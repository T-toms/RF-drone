import os,json,re,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.io import loadmat
from scipy import ndimage as ndi,fft
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path('C:/Users/Administrator/Desktop/数据集处理'); OUT=ROOT/'work/burst_regularity_pilot_20261106'; RAW=Path('G:/drone')
RECS=['007_5.8GHz_20M-signal__2026-09-20-20-34-08','002_5.8GHz_10M-signal__2026-09-20-19-40-26','F005_5.8GHz_40M-signal__2026-09-20-20-12-14','005_5.8GHz_10M-signal__2026-09-20-20-14-22','012_2.4GHz_10M-signal__2026-09-20-18-58-55','F009_2.4GHZ_20M-signal__2026-09-20-18-50-14','20260709-025842_mini3pro','20260709-021044_ambient']
NFFT=1024;HOP=512;THRESH=10;EPS=.45;MIN_SAMPLES=5
RULES=dict(persistent=.9,periodic_n=20,periodic_reg=.7,duration_cv=.3,hop_channels=3,spacing=.6,visits=3,bandwidth_cv=.3,isolated_n=5,isolated_reg=.4)
def dump(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')
def snapshot():
    n=0;latest=0
    for base,dirs,files in os.walk(RAW):
        for f in files:
            s=(Path(base)/f).stat();n+=1;latest=max(latest,s.st_mtime_ns)
    return dict(file_count=n,latest_mtime_ns=latest)
def stft(x,fs):
    w=np.hanning(1024).astype('float32'); a=np.lib.stride_tricks.sliding_window_view(x,1024)[::512]
    p=abs(fft.fftshift(fft.fft(a*w,axis=1,workers=2),axes=1))**2/sum(w*w)
    return 10*np.log10(np.maximum(p,1e-20))
def period(t,b):
    gaps=np.concatenate([np.diff(np.sort(t[b==k])) for k in np.unique(b)]) if len(t)>1 else np.array([])
    gaps=gaps[gaps>0.05]
    if not len(gaps):return 0.,0.,[]
    h=np.zeros(4001)
    for block in np.unique(b):
        z=t[b==block]; bins=np.bincount(np.rint((z-z.min())/.05).astype(int)); size=fft.next_fast_len(2*len(bins))
        ac=fft.irfft(abs(fft.rfft(bins,n=size))**2,n=size)[:min(4001,len(bins))];h[:len(ac)]+=ac
    h[:2]=0;inds=np.argsort(h)[-30:]; candidates=list(inds*.05)+list(np.quantile(gaps,np.linspace(0,1,30)))
    best=(0.,0.)
    for c in candidates:
        if c<.1:continue
        k=np.maximum(1,np.rint(gaps/c)); score=np.mean(abs(gaps-k*c)<=.03*c)
        if score>best[0]+.001 or (abs(score-best[0])<.001 and c>best[1]):best=(float(score),float(c))
    return best[1],best[0],gaps.tolist()
def intervals_union(items):
    z=[]
    for a,b in sorted(items):
        if z and a<=z[-1][1]:z[-1][1]=max(z[-1][1],b)
        else:z.append([float(a),float(b)])
    return z
def process(rec,index):
    cp=OUT/(rec+'.json')
    if cp.exists():return json.loads(cp.read_text(encoding='utf8'))
    print('START',rec,flush=True); chunks=[]; isv=index>=6
    if not isv:
        folder,session=rec.split('__'); tab=pd.read_csv(ROOT/'work/audit_drone_20261023/data_preprocessing_manifest.csv',dtype=str).fillna('')
        rows=tab[(tab.folder==folder)&(tab.session==session)&(tab.duplicate=='False')&(tab.analysis_eligible=='True')].copy();rows['ord']=rows.epoch.astype(int)*20+rows.seg.astype(int); rows=rows.sort_values('ord')
        fs=122880000.;fc=2400. if '2.4' in folder else 5800.;floor=json.loads((Path('E:/drone_lab_cache_20261031')/rec/'meta.json').read_text())['floor_db']
        for _,r in rows.iterrows():chunks.append(dict(paths=[str(RAW/r.path)],start=int(r.ord)*50.,block=0,epoch=int(r.epoch),lo=int(r.start_sample),hi=int(r.end_sample_exclusive)))
    else:
        files=sorted((RAW/'datasets_v3').rglob('*'+rec+'*.mat')); groups={}
        for p in files:groups.setdefault(int(re.search(r'_(\d{6})_s',p.name)[1]),[]).append(str(p))
        fs=float(loadmat(files[0],variable_names=['fs'])['fs'].item());fc=float(re.search(r'_(\d+)M_25dB',files[0].name)[1]);floor=None;offset=0
        for block,paths in sorted(groups.items()):
            size=sum(loadmat(p,variable_names=['IQ'])['IQ'].size for p in paths)
            chunks.append(dict(paths=paths,start=offset,block=block,epoch=block,lo=0,hi=size));offset+=size/fs*1000
    freq=fc+fft.fftshift(fft.fftfreq(1024,1/fs))/1e6; valid=abs(freq-fc)<=fs*.45/1e6
    tones=pd.read_csv(ROOT/'work/prep_lines_20261104/lines_classified.csv');tones=tones[(tones.recording==rec)&(tones.classification=='persistent_tone_candidate')]
    for f in tones.freq_mhz: valid &= abs(freq-f)>.15
    if isv:
        floor_cells=[]
        for c in chunks:
            x=np.concatenate([loadmat(p,variable_names=['IQ'])['IQ'].ravel().astype('complex64') for p in c['paths']])
            pp=stft(x,fs); groups=np.array_split(pp,max(1,round(len(x)/fs*1000)))
            floor_cells.extend([10*np.log10(np.maximum(np.mean(10**(q/10),axis=0),1e-20)) for q in groups])
        floor=float(np.median(np.percentile(floor_cells,10,axis=0)[valid]))
    rng=np.random.default_rng(20261106); selections=sorted(rng.choice(len(chunks),3,replace=False).tolist()); events=[]; views=[]; coarse=[]; boundaries=[]
    for ci,c in enumerate(chunks):
        x=np.concatenate([loadmat(p,variable_names=['IQ' if isv else 'data'])['IQ' if isv else 'data'].ravel().astype('complex64') for p in c['paths']])[c['lo']:c['hi']]
        p=stft(x,fs);dt=512/fs*1000;start=c['start']+c['lo']/fs*1000
        if floor is None:
            # Task A: per-bin 10th percentile of 1ms linear-mean powers, then median within mask.
            cells=[10*np.log10(np.maximum(np.mean(10**(q/10),axis=0),1e-20)) for q in np.array_split(p,max(1,round(len(x)/fs*1000)))]; floor=float(np.median(np.percentile(cells,10,axis=0)[valid]))
        mask=ndi.binary_closing((p>floor+10)&valid[None,:],structure=np.ones((3,2)),border_value=0)
        # Closing with padded edges preserves boundary contact for cross-segment union.
        raw=(p>floor+10)&valid[None,:]; mask=ndi.binary_closing(np.pad(raw,((2,2),(1,1)),mode='edge'),structure=np.ones((3,2)))[2:-2,1:-1];mask &=valid[None,:]
        lab,n=ndi.label(mask,np.ones((3,3))); objects=ndi.find_objects(lab)
        current=[]
        for k,sl in enumerate(objects,1):
            a,b=sl;local=(lab[sl]==k)
            if local.sum()<6:continue
            runs=[(0,local.shape[0])]
            if local.shape[0]*dt>20:
                occupied=local.mean(axis=1)>=.5; ll,nn=ndi.label(occupied);runs=[(v[0].start,v[0].stop) for v in ndi.find_objects(ll)]
            for lo,hi in runs:
                sub=local[lo:hi];pix=p[a.start+lo:a.start+hi,b][sub]
                if sub.sum()<6:continue
                ys=np.where(sub.any(axis=0))[0]; flo=freq[b.start+ys.min()]-fs/1024/2e6;fhi=freq[b.start+ys.max()]+fs/1024/2e6
                ev=dict(t_start_ms=start+(a.start+lo)*dt,t_end_ms=start+(a.start+hi)*dt+dt,f_lo_MHz=flo,f_hi_MHz=fhi,peak_db_above_floor=float(pix.max()-floor),mean_db_above_floor=float(10*np.log10(np.mean(10**((pix-floor)/10)))),area=int(sub.sum()),epoch=c['epoch'],block=c['block'],parts=1)
                # Join components at physically adjacent file boundaries only.
                matches=[e for e in events if e['block']==ev['block'] and abs(e['t_end_ms']-ev['t_start_ms'])<3*dt and min(e['f_hi_MHz'],fhi)>=max(e['f_lo_MHz'],flo)-fs/1024/1e6] if a.start+lo<=2 else []
                if matches:
                    e=matches[-1];old=e['area'];e.update(t_end_ms=ev['t_end_ms'],f_lo_MHz=min(e['f_lo_MHz'],flo),f_hi_MHz=max(e['f_hi_MHz'],fhi),peak_db_above_floor=max(e['peak_db_above_floor'],ev['peak_db_above_floor']),mean_db_above_floor=float(10*np.log10((10**(e['mean_db_above_floor']/10)*old+10**(ev['mean_db_above_floor']/10)*ev['area'])/(old+ev['area']))),area=old+ev['area'],parts=e['parts']+1);boundaries.append(dict(boundary_ms=start,event_start=e['t_start_ms']))
                else:events.append(ev)
        coarse.extend([(start+j*1.,np.mean(p[np.floor(np.arange(len(p))*dt).astype(int)==j],axis=0)) for j in range(math.floor(len(x)/fs*1000))])
        if ci in selections:
            np.savez_compressed(OUT/f'view_{index+1}_{ci}.npz',p=p[:round(50/dt)],start=start,dt=dt,freq=freq,floor=floor)
            views.append(dict(chunk=ci,start_ms=start,end_ms=min(start+50,start+len(x)/fs*1000)))
        print(rec,ci+1,'/',len(chunks),'events',len(events),flush=True)
    pd.DataFrame(events).to_csv(OUT/f'raw_events_{rec}.csv',index=False)
    df=pd.DataFrame(events)
    df['duration_ms']=df.t_end_ms-df.t_start_ms;df['bandwidth_MHz']=df.f_hi_MHz-df.f_lo_MHz;df['f_center_MHz']=(df.f_lo_MHz+df.f_hi_MHz)/2
    feat=StandardScaler().fit_transform(np.c_[np.log(df.bandwidth_MHz),np.log(df.duration_ms),df.mean_db_above_floor]);grid,inverse,weights=np.unique(np.round(feat/.05)*.05,axis=0,return_inverse=True,return_counts=True); labels=DBSCAN(eps=EPS,min_samples=MIN_SAMPLES).fit(grid,sample_weight=weights).labels_[inverse]
    # Noise events remain individual signals, never combine unrelated DBSCAN outliers.
    noise=np.where(labels<0)[0];labels[noise]=np.arange(len(noise))+max(labels.max()+1,0);df['signal_id']=labels
    signals=[];total_end=max(c['start']+(c['hi']-c['lo'])/fs*1000 for c in chunks)
    for sid,g in df.groupby('signal_id'):
        chans=[];ids={}
        for ix,e in g.sort_values('f_center_MHz').iterrows():
            cand=[j for j,z in enumerate(chans) if abs(e.f_center_MHz-z['center'])<.5*min(e.bandwidth_MHz,z['bw'])]
            if cand:j=cand[-1];chans[j]['count']+=1
            else:j=len(chans);chans.append(dict(center=float(e.f_center_MHz),bw=float(e.bandwidth_MHz),count=1))
            ids[ix]=j
        df.loc[g.index,'channel_id']=[ids[i] for i in g.index];spacing=np.diff([z['center'] for z in chans]);sm=0.;sr=0.
        if len(spacing):
            vals=np.round(spacing/(fs/1024/1e6));uniq,counts=np.unique(vals,return_counts=True);sm=float(uniq[counts.argmax()]*fs/1024/1e6);sr=float(counts.max()/len(spacing))
        t=g.t_start_ms.to_numpy(); b=g.block.to_numpy();T,reg,iat=period(t,b);cvd=float(g.duration_ms.std(ddof=0)/g.duration_ms.mean());cvb=float(g.bandwidth_MHz.std(ddof=0)/g.bandwidth_MHz.mean())
        rates={}
        exposure={}
        for c in chunks:
            exposure.setdefault(c['block'],[]).append((c['start']+c['lo']/fs*1000,c['start']+c['hi']/fs*1000))
        for win in [10,20,30,40,50,100]:
            hits=[]
            for block,ranges in exposure.items():
                for low,high in intervals_union(ranges):
                    starts=np.arange(low,high-1e-6,win); ends=np.minimum(starts+win,high)
                    gg=g[g.block==block]; hit=np.zeros(len(starts),bool)
                    for row in gg.itertuples():hit|=(row.t_start_ms<ends)&(row.t_end_ms>starts)
                    hits.extend(hit.tolist())
            rates[win]=float(np.mean(hits))
        eps=sorted(g.epoch.unique());both=len(eps)>=2;reasons=[]
        if rates[50]>=.9:reasons.append('a')
        if len(g)>=20 and reg>=.7 and cvd<=.3 and both:reasons.append('b')
        if len(chans)>=3 and sr>=.6 and min(z['count'] for z in chans)>=3 and len(g)>=20 and cvb<=.3 and both:reasons.append('c')
        category='featured' if reasons else ('isolated' if len(g)<5 or (reg<.4 and sr<.4) else 'review')
        epochstats={str(ep):dict(n=int(sum(g.epoch==ep)),T0_ms=period(g[g.epoch==ep].t_start_ms.to_numpy(),g[g.epoch==ep].block.to_numpy())[0],channels=sorted(set(ids[i] for i in g[g.epoch==ep].index))) for ep in eps}
        channelstats={str(j):dict(T0_ms=period(g.loc[[i for i in g.index if ids[i]==j]].t_start_ms.to_numpy(),g.loc[[i for i in g.index if ids[i]==j]].block.to_numpy())[0],time_regularity=period(g.loc[[i for i in g.index if ids[i]==j]].t_start_ms.to_numpy(),g.loc[[i for i in g.index if ids[i]==j]].block.to_numpy())[1],**z) for j,z in enumerate(chans)}
        signals.append(dict(recording=rec,signal_id=int(sid),n_events=len(g),f_lo_MHz=float(g.f_lo_MHz.min()),f_hi_MHz=float(g.f_hi_MHz.max()),bandwidth_MHz=float(g.bandwidth_MHz.median()),duration_ms=float(g.duration_ms.median()),T0_ms=T,time_regularity=reg,n_channels=len(chans),spacing_MHz=sm,spacing_regularity=sr,cv_duration=cvd,cv_bandwidth=cvb,both_epochs=both,epoch_metrics=epochstats,channel_metrics=channelstats,epoch_channels_consistent=len({str(z['channels']) for z in epochstats.values()})==1,epoch_T0_consistent=(max(z['T0_ms'] for z in epochstats.values())-min(z['T0_ms'] for z in epochstats.values())<=.03*max(T,.001)),coverage_100ms=rates[100],window_presence={str(k):v for k,v in rates.items()},classification=category,rule='/'.join(reasons),iat_ms=iat))
    df.to_csv(OUT/f'bursts_{rec}.csv',index=False)
    np.savez_compressed(OUT/f'overview_{index+1}.npz',times=np.array([x[0] for x in coarse]),power=np.array([x[1] for x in coarse]),freq=freq,floor=floor)
    result=dict(recording=rec,index=index+1,signals=signals,views=views,boundaries=boundaries,floor_db=floor,fs=fs,duration_ms=total_end,tones_removed=tones.freq_mhz.tolist());dump(cp,result);return result
if __name__=='__main__':
    before=OUT/'source_before.json'
    if not before.exists():dump(before,snapshot())
    results=[process(r,i) for i,r in enumerate(RECS)]
    dump(OUT/'source_after.json',snapshot());dump(OUT/'results.json',results)
    print('COMPUTE COMPLETE',flush=True)
