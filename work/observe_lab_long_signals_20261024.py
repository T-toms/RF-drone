"""Read-only long-recording signal overview in sample/normalized-frequency units."""
from pathlib import Path
import csv,json,hashlib,warnings
import numpy as np
from scipy.io import loadmat
from scipy.fft import fft,fftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W=Path(__file__).parent;A=W/'audit_drone_20261023';O=W/'long_signal_pilot_20261024';O.mkdir(exist_ok=True)
ROOT=Path('G:/drone');SEG=6144000;NFFT=4096;GROUP=15;BIN=NFFT*GROUP;FREQ=1024
PILOTS=['009_2.4GHZ_10M-signal','009_2.4GHZ_60M-signal','007_5.8GHz_10M-signal','F009_2.4GHZ_60M-signal']
rows=list(csv.DictReader((A/'data_preprocessing_manifest.csv').open(encoding='utf-8-sig')))
selected=[]
for folder in PILOTS:
    candidates=[r for r in rows if r['folder']==folder and r['duplicate']=='False']
    session=min(r['session'] for r in candidates)
    selected.append(dict(folder=folder,session=session,files=sorted([r for r in candidates if r['session']==session],key=lambda r:(int(r['epoch']),int(r['seg'])))))
(O/'selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf8')
summaries=[];win=np.hanning(NFFT).astype(np.float32);wn=float(np.sum(win**2))
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#bdbdbd')

def profile(z,nfft):
    w=np.hanning(nfft).astype(np.float32)
    return np.abs(fftshift(fft(z*w,axis=1),axes=1))**2/np.sum(w*w)
def db(z):return 10*np.log10(np.maximum(z,1e-25))

for record in selected:
    tag=record['folder']+'__'+record['session'];complete=O/(tag+'_summary.json')
    if complete.exists():summaries.append(json.loads(complete.read_text(encoding='utf8')));continue
    mean=np.full((4000,FREQ),np.nan,dtype=np.float32);peak=np.full_like(mean,np.nan)
    envelope=np.full(4000,np.nan,dtype=np.float32);envmax=np.full_like(envelope,np.nan)
    spectrum=np.zeros(NFFT,dtype=np.float64);validframes=0;validcounts=np.zeros(4000,dtype=np.int16)
    source=[]
    for j,r in enumerate(record['files']):
        ordinal=int(r['epoch'])*20+int(r['seg']);source.append(dict(path=r['path'],ordinal=ordinal,eligible=r['analysis_eligible']=='True',valid_start=r['start_sample'],valid_end=r['end_sample_exclusive']))
        if r['analysis_eligible']!='True':continue
        x=loadmat(ROOT/r['path'],variable_names=['data'])['data'].ravel();assert len(x)==SEG
        frames=x.reshape(-1,NFFT);start,end=int(r['start_sample']),int(r['end_sample_exclusive'])
        offsets=np.arange(len(frames))*NFFT;ok=(offsets>=start)&(offsets+NFFT<=end)
        p=profile(frames,NFFT);spectrum+=p[ok].sum(axis=0,dtype=np.float64);validframes+=int(ok.sum())
        e=np.mean(np.abs(frames)**2,axis=1);p[~ok]=0;e[~ok]=np.nan
        shaped=p.reshape(100,GROUP,NFFT);count=ok.reshape(100,GROUP).sum(axis=1);validcounts[ordinal*100:(ordinal+1)*100]=count
        ave=shaped.sum(axis=1)/np.maximum(count[:,None],1);mx=shaped.max(axis=1)
        ave[count==0]=np.nan;mx[count==0]=np.nan
        sl=slice(ordinal*100,(ordinal+1)*100)
        mean[sl]=db(ave.reshape(100,FREQ,4).mean(axis=2));peak[sl]=db(mx.reshape(100,FREQ,4).max(axis=2))
        ee=e.reshape(100,GROUP);keep=count>0
        v=np.full(100,np.nan,dtype=np.float32);v[keep]=np.nansum(ee[keep],axis=1)/count[keep];envelope[sl]=db(v)
        v=np.full(100,np.nan,dtype=np.float32);v[keep]=np.nanmax(ee[keep],axis=1);envmax[sl]=db(v)
        if j%5==0:print(record['folder'],j+1,'/',len(record['files']),flush=True)
    assert validframes>0
    spec=spectrum/validframes;ref=float(np.nanpercentile(mean,99.5));eref=float(np.nanmedian(envelope))
    missing=sorted(set(range(40))-{int(r['epoch'])*20+int(r['seg']) for r in record['files']})
    fig,axs=plt.subplots(4,1,figsize=(15,11),gridspec_kw={'height_ratios':[1.1,2.1,2.1,1.1]},layout='constrained')
    f=np.arange(NFFT)/NFFT-.5;axs[0].plot(f,db(spec)-np.max(db(spec)),lw=.8);axs[0].set(xlim=(-.5,.5),ylim=(-65,3),xlabel='Normalized frequency (cycles/sample)',ylabel='PSD relative to peak (dB)');axs[0].grid(alpha=.25)
    extent=[0,40*SEG/1e6,-.5,.5]
    for ax,z,title in [(axs[1],mean,'Mean spectrum in each time bin'),(axs[2],peak,'Maximum spectrum in each time bin (shows sparse energy)')]:
        im=ax.imshow((z-ref).T,aspect='auto',origin='lower',extent=extent,cmap=cmap,vmin=-45,vmax=10,interpolation='nearest')
        ax.set_ylabel('Cycles/sample');ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,label='Relative PSD (dB)')
        ax.axvline(20*SEG/1e6,color='white',ls='--',lw=.8,alpha=.7)
    t=(np.arange(4000)+.5)*BIN/1e6;axs[3].plot(t,envelope-eref,lw=.7,label='Mean short-frame power');axs[3].plot(t,envmax-eref,lw=.5,alpha=.65,label='Maximum short-frame power')
    axs[3].set(xlabel='Original recording sample index (million)',ylabel='Power relative to median (dB)',xlim=(0,40*SEG/1e6));axs[3].legend(fontsize=8);axs[3].grid(alpha=.25)
    fig.suptitle(record['folder']+' / '+record['session']+'\nSampling rate not assumed; gray = missing or excluded numeric data',fontsize=13)
    fig.savefig(O/(tag+'_overview.png'),dpi=130);plt.close(fig)
    # Choose an intermittent/variable region objectively; this is not a label.
    variation=np.full(4000,-np.inf,dtype=np.float32);covered=validcounts>0
    variation[covered]=np.percentile((peak-mean)[covered],95,axis=1)
    selectedbin=int(np.argmax(variation));ordinal=selectedbin//100
    r=next(r for r in record['files'] if int(r['epoch'])*20+int(r['seg'])==ordinal)
    x=loadmat(ROOT/r['path'],variable_names=['data'])['data'].ravel();center=(selectedbin%100)*BIN+BIN//2;length=262144
    lo=max(int(r['start_sample']),min(center-length//2,int(r['end_sample_exclusive'])-length));hi=min(lo+length,int(r['end_sample_exclusive']))
    z=x[lo:hi];nn=2048;hop=512;views=np.lib.stride_tricks.sliding_window_view(z,nn)[::hop];pp=profile(views,nn)
    zr=float(np.percentile(db(pp),99.5));fig,axs=plt.subplots(2,1,figsize=(14,7),layout='constrained',gridspec_kw={'height_ratios':[3,1]})
    im=axs[0].imshow(db(pp).T-zr,aspect='auto',origin='lower',extent=[0,len(z)/1e6,-.5,.5],cmap='viridis',vmin=-45,vmax=10,interpolation='nearest');fig.colorbar(im,ax=axs[0],label='Relative PSD (dB)');axs[0].set_ylabel('Cycles/sample')
    pe=np.mean(np.abs(views)**2,axis=1);axs[1].plot((np.arange(len(pe))*hop+nn/2)/1e6,db(pe)-np.median(db(pe)),lw=.6);axs[1].set(xlabel='Samples from local start (million)',ylabel='Relative power (dB)');axs[1].grid(alpha=.25)
    fig.suptitle(record['folder']+' — local region with high spectral variation\nOriginal sample interval ['+str(ordinal*SEG+lo)+', '+str(ordinal*SEG+hi)+')',fontsize=12)
    fig.savefig(O/(tag+'_zoom.png'),dpi=130);plt.close(fig)
    np.savez_compressed(O/(tag+'_statistics.npz'),mean_db=mean,maximum_db=peak,envelope_db=envelope,envelope_max_db=envmax,mean_psd=spec.astype(np.float32),valid_frame_counts=validcounts)
    summary=dict(folder=record['folder'],session=record['session'],files=len(record['files']),eligible_files=sum(r['analysis_eligible']=='True' for r in record['files']),missing_ordinals=missing,
        valid_fft_frames=validframes,fft_samples_processed=validframes*NFFT,fft_size=NFFT,fft_hop=NFFT,time_bin_samples=BIN,frames_per_bin=GROUP,
        normalized_frequency=True,sampling_rate_assumed=None,center_frequency_assumed=None,mean_psd_reference_db=ref,envelope_p05_p50_p95_db=[float(v) for v in np.nanpercentile(envelope,[5,50,95])],
        positive_frequency_power_fraction=float(spec[NFFT//2:].sum()/spec.sum()),overview=tag+'_overview.png',zoom=tag+'_zoom.png',
        zoom_source=r['path'],zoom_source_interval=[lo,hi],zoom_recording_interval=[ordinal*SEG+lo,ordinal*SEG+hi],source=source)
    complete.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');summaries.append(summary)
    print('COMPLETE',record['folder'],flush=True)
(O/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')
print('Pilot observation complete',flush=True)
