from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pilot import *
EPS=.07

def redo(r):
    rec=r['recording'];cp=OUT/(rec+'.json');index=r['index']-1;fs=r['fs'];floor=r['floor_db'];views=r['views'];boundaries=r['boundaries']
    tones=pd.DataFrame({'freq_mhz':r['tones_removed']})
    try:events=pd.read_csv(OUT/f'raw_events_{rec}.csv').to_dict('records')
    except pd.errors.EmptyDataError:events=[]
    if index<6:
        folder,session=rec.split('__')
        tab=pd.read_csv(ROOT/'work/audit_drone_20261023/data_preprocessing_manifest.csv',dtype=str).fillna('')
        rows=tab[(tab.folder==folder)&(tab.session==session)&(tab.duplicate=='False')&(tab.analysis_eligible=='True')]
        chunks=[dict(start=(int(z.epoch)*20+int(z.seg))*50.,lo=int(z.start_sample),hi=int(z.end_sample_exclusive),block=0,epoch=int(z.epoch)) for z in rows.itertuples()]
    else:
        files=sorted((RAW/'datasets_v3').rglob('*'+rec+'*.mat')); groups={}
        for path in files:groups.setdefault(int(re.search(r'_(\d{6})_s',path.name)[1]),[]).append(path)
        chunks=[];offset=0
        for block,paths in sorted(groups.items()):
            size=sum(loadmat(path,variable_names=['IQ'])['IQ'].size for path in paths)
            chunks.append(dict(start=offset,lo=0,hi=size,block=block,epoch=block));offset+=size/fs*1000
    events=merge_fragments(events)
    df=pd.DataFrame(events)
    if df.empty:
        df=pd.DataFrame(columns=['t_start_ms','t_end_ms','f_lo_MHz','f_hi_MHz','area','energy','n_merged','block','epoch','parts','mean_db_above_floor','peak_db_above_floor'])
    df['duration_ms']=df.t_end_ms-df.t_start_ms;df['bandwidth_MHz']=df.f_hi_MHz-df.f_lo_MHz;df['f_center_MHz']=(df.f_lo_MHz+df.f_hi_MHz)/2
    cutoff=float(df.energy.quantile(.05)) if len(df) else 0
    df['fragment']=(df.duration_ms<.05)|(df.bandwidth_MHz<.3)|(df.energy<cutoff)
    df['signal_id']=-1;df['channel_id']=-1
    good=df[~df.fragment].copy()
    if len(good):
        feat=StandardScaler().fit_transform(np.c_[np.log(good.bandwidth_MHz),np.log(good.duration_ms)])
        labels=DBSCAN(eps=EPS,min_samples=MIN_SAMPLES).fit_predict(feat)
        noise=np.where(labels<0)[0];labels[noise]=np.arange(len(noise))+max(labels.max()+1,0)
        df.loc[good.index,'signal_id']=labels
    signals=[];total_end=max(c['start']+(c['hi']-c['lo'])/fs*1000 for c in chunks)
    for sid,g in df[~df.fragment].groupby('signal_id'):
        chans=[];ids={}
        for ix,e in g.sort_values('f_center_MHz').iterrows():
            cand=[j for j,z in enumerate(chans) if abs(e.f_center_MHz-z['center'])<.5*min(e.bandwidth_MHz,z['bw'])]
            if cand:j=cand[-1];chans[j]['count']+=1
            else:j=len(chans);chans.append(dict(center=float(e.f_center_MHz),bw=float(e.bandwidth_MHz),count=1))
            ids[ix]=j
        df.loc[g.index,'channel_id']=[ids[i] for i in g.index];
        for j,z in enumerate(chans):
            members=g.loc[[i for i in g.index if ids[i]==j]]
            z['center']=float(members.f_center_MHz.median());z['bw']=float(members.bandwidth_MHz.median())
        spacing=np.diff(sorted(z['center'] for z in chans));sm=0.;sr=0.
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
                    gg=g[g.block==block]; areas=np.zeros(len(starts))
                    for row in gg.itertuples():
                        overlap=np.maximum(0,np.minimum(row.t_end_ms,ends)-np.maximum(row.t_start_ms,starts))
                        areas+=overlap/row.duration_ms*row.area*512/fs*1000*(fs/1024/1e6)
                    hits.extend((areas>=.5*np.median(g.area*512/fs*1000*(fs/1024/1e6))).tolist())
            rates[win]=float(np.mean(hits))
        eps=sorted(g.epoch.unique());both=len(eps)>=2;reasons=[]
        if rates[50]>=.9:reasons.append('a')
        if len(g)>=20 and reg>=.7 and cvd<=.3 and both:reasons.append('b')
        if len(chans)>=3 and sr>=.6 and min(z['count'] for z in chans)>=3 and len(g)>=20 and cvb<=.3 and both:reasons.append('c')
        category='featured' if reasons else ('isolated' if len(g)<5 or (reg<.4 and sr<.4) else 'review')
        iat_cv=float(np.std(iat)/np.mean(iat)) if len(iat) and np.mean(iat)>0 else None
        sparse=5<=len(g)<=19 and iat_cv is not None and iat_cv<=.2
        if sparse:category='review';reasons=['sparse_periodic']
        seq=hop_sequence(g,ids,len(chans),reasons)
        epochstats={str(ep):dict(n=int(sum(g.epoch==ep)),T0_ms=period(g[g.epoch==ep].t_start_ms.to_numpy(),g[g.epoch==ep].block.to_numpy())[0],channels=sorted(set(ids[i] for i in g[g.epoch==ep].index))) for ep in eps}
        channelstats={str(j):dict(T0_ms=period(g.loc[[i for i in g.index if ids[i]==j]].t_start_ms.to_numpy(),g.loc[[i for i in g.index if ids[i]==j]].block.to_numpy())[0],time_regularity=period(g.loc[[i for i in g.index if ids[i]==j]].t_start_ms.to_numpy(),g.loc[[i for i in g.index if ids[i]==j]].block.to_numpy())[1],**z) for j,z in enumerate(chans)}
        signals.append(dict(recording=rec,signal_id=int(sid),**seq,iat_cv=iat_cv,sparse_periodic=sparse,mean_db_above_floor=float(g.mean_db_above_floor.median()),n_events=len(g),f_lo_MHz=float(g.f_lo_MHz.min()),f_hi_MHz=float(g.f_hi_MHz.max()),bandwidth_MHz=float(g.bandwidth_MHz.median()),duration_ms=float(g.duration_ms.median()),T0_ms=T,time_regularity=reg,n_channels=len(chans),spacing_MHz=sm,spacing_regularity=sr,cv_duration=cvd,cv_bandwidth=cvb,both_epochs=both,epoch_metrics=epochstats,channel_metrics=channelstats,epoch_channels_consistent=len({str(z['channels']) for z in epochstats.values()})==1,epoch_T0_consistent=(max(z['T0_ms'] for z in epochstats.values())-min(z['T0_ms'] for z in epochstats.values())<=.03*max(T,.001)),coverage_100ms=rates[100],window_presence={str(k):v for k,v in rates.items()},classification=category,rule='/'.join(reasons),iat_ms=iat))
    df.to_csv(OUT/f'bursts_{rec}.csv',index=False)

    result=dict(recording=rec,index=index+1,signals=signals,views=views,boundaries=boundaries,floor_db=floor,fs=fs,duration_ms=total_end,tones_removed=tones.freq_mhz.tolist(),fragment_count=int(df.fragment.sum()),energy_percentile_5=cutoff);dump(cp,result);return result

if __name__=='__main__':
    results=[redo(r) for r in json.loads((OUT/'results.json').read_text(encoding='utf8'))]
    dump(OUT/'results.json',results)
    print('RECLUSTER COMPLETE',flush=True)
