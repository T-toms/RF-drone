"""Q pilot. All source opens are read only; run using D:/anaconda/python.exe -I.
Matrices are time x frequency. Intervals in all CSVs are half open.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONIOENCODING'] = 'utf-8'
from pathlib import Path
import json, csv, hashlib, collections, time, math, sys
import numpy as np
import pandas as pd
import psutil
from scipy import io, fft, ndimage as ndi

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = Path('G:/drone')
CACHE = Path('E:/side_Q_band_detect_pilot_20261111')
PARAMS = [(k,g) for k in (6,10) for g in (0.5,2)]
PROTECTED = ['crop_slice_20261107','side_M_notchfill_20261110','main_W_window_20261110']
PROC = psutil.Process()
peak = 0

def dump(name, x):
    p=OUT/name; tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)

def check_resources():
    global peak
    peak = max(peak, PROC.memory_info().peak_wset)
    assert peak < 8*1024**3, ('memory limit',peak)
    assert sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()) <= 10*10**9

def snapshot():
    d = {}
    for name in PROTECTED:
        assert (ROOT/'work'/name).is_dir()
        for p in sorted((ROOT/'work'/name).rglob('*')):
            if p.is_file():
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
                d[str(p.relative_to(ROOT))] = {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    return d

def scalar(x):
    return '|'.join('unknown' if v is None else str(v) for v in x) if isinstance(x,list) and x else (str(x) if x else 'unknown')

def inventory():
    # Use old plan ONLY for descriptive labels; never use its crop or split to select samples.
    plan = json.loads((ROOT/'work/crop_slice_20261107/plan.json').read_text(encoding='utf-8'))
    labels = {b['group_id']:b for b in plan if b['dataset']=='data'}
    runs = json.loads((ROOT/'work/audit_drone_20261023/data_usable_runs.json').read_text(encoding='utf-8'))['runs']
    groups = collections.defaultdict(list)
    for r in runs: groups[(r['folder'],r['session'])].append(r)
    clips=[]
    for (folder,session),rs in sorted(groups.items()):
        gid='data:'+folder+'__'+session
        b=labels[gid]; fs=122880000; length=round(fs*.020)
        rs=sorted(rs,key=lambda r:r['start_sample'])
        # Six starts uniformly spaced in the union of valid START intervals.
        spans=[(r,max(0,r['end_sample_exclusive']-r['start_sample']-length+1)) for r in rs]
        total=sum(n for r,n in spans); assert total>0
        for q,rank in enumerate(np.linspace(0,total-1,6).round().astype(np.int64)):
            for r,n in spans:
                if rank<n: break
                rank-=n
            start=r['start_sample']+int(rank)
            parts=[]
            for p in r['parts']:
                a=p['recording_sample_start']; z=a+p['end_sample_exclusive']-p['start_sample']
                lo=max(start,a); hi=min(start+length,z)
                if hi>lo:
                    parts.append({'path':str(DATA/p['path']),'start':p['start_sample']+lo-a,'count':hi-lo})
            assert sum(p['count'] for p in parts)==length
            clips.append(dict(source='data',band=b['band'],model=b['model'],state=b['state'],bw=b['bw_label'],label='drone',
                              group=gid,date=session[:10],fs=fs,fc=float(b['fc_in']),start=start,length=length,parts=parts,slot=q))
    exclusions=[]
    for p in sorted((DATA/'raw').rglob('*.json')):
        m=json.loads(p.read_text(encoding='utf-8')); c=m['capture']; d=m['declared']; fs=c['sample_rate_hz']; fc=c['center_freq_hz']/1e6
        reason=''
        if p.parent.parent==DATA/'raw': reason='duplicate_root_session'
        elif fs!=250000000: reason='125MSps_sweep'
        elif '20260709-030521_ambient' in str(p): reason='short_sweep'
        elif fc not in (2440,2450,2500,5800): reason='outside_target_centers'
        if reason:
            exclusions.append({'path':str(p),'reason':reason}); continue
        assert c['datatype']=='ci16_le' and c['sample_count']==30000000, str(p)
        bp=p.with_suffix('.bin'); assert bp.stat().st_size==c['sample_count']*4
        for slot,start in enumerate((0,(c['sample_count']-round(fs*.02))//2)):
            clips.append(dict(source='raw',band='2.4GHz' if fc<3000 else '5.8GHz',model=scalar(d['model']),state=scalar(d['state']),
                              bw=scalar(d['bw']),label=d['class'],group=m['session']['id'],date=c['t_start_utc'][:10],
                              fs=fs,fc=fc,start=start,length=round(fs*.02),parts=[dict(path=str(bp),start=start,count=round(fs*.02))],slot=slot))
    for i,c in enumerate(clips): c['clip_id']=f'Q{i:04d}'
    dump('clips.json',clips); pd.DataFrame(exclusions).to_csv(OUT/'excluded.csv',index=False,encoding='utf-8-sig')
    return clips

def read_iq(c):
    arrays=[]
    for p in c['parts']:
        path=Path(p['path']); assert path.drive.upper()=='G:'
        if c['source']=='data':
            # One bounded 50-ms MAT container at a time (49 MB), not the recording.
            a=io.loadmat(path,variable_names=['data'])['data'].ravel()
            arrays.append(a[p['start']:p['start']+p['count']].astype(np.complex64,copy=True))
            del a
        else:
            with path.open('rb') as f:
                f.seek(p['start']*4); a=np.fromfile(f,dtype='<i2',count=p['count']*2)
            assert len(a)==p['count']*2
            arrays.append(a[::2].astype(np.float32)+1j*a[1::2].astype(np.float32))
    x=np.concatenate(arrays) if len(arrays)>1 else arrays[0]
    assert len(x)==c['length']; return x

def spectrum(c):
    n=1024 if c['source']=='data' else 2048
    fs=c['fs']; fc=c['fc']; x=read_iq(c); nt=len(x)//n
    a=x[:nt*n].reshape(nt,n)*np.hanning(n).astype(np.float32)
    z=fft.fftshift(fft.fft(a,axis=1,workers=1),axes=1)
    power=z.real*z.real+z.imag*z.imag
    f=fc+fft.fftshift(fft.fftfreq(n,1/(fs/1e6)))
    bounds=(2395,2488.5) if fc<3000 else (5720,5855)
    lo=max(bounds[0],fc-fs/2e6+1.5); hi=min(bounds[1],fc+fs/2e6-1.5)
    ix=(f>=lo)&(f<=hi); f=f[ix]; power=power[:,ix]
    db=10*np.log10(np.maximum(power,1e-30))
    tones=[fc,2426.88,2457.6,5740,5795.5] if c['source']=='data' else [fc]
    bad=np.zeros(len(f),bool)
    for tone in tones: bad |= abs(f-tone)<=.3
    # Tone columns are excluded from noise estimation and assigned F; avoids circular definition.
    floor=float(np.percentile(db[:,~bad],20)); db[:,bad]=floor
    # Raw reference follows outside_crop_v3: 1-ms maximum, floor from time-mean power.
    per=int(round(fs/1000))//n; nc=nt//per
    bpower=power[:nc*per].reshape(nc,per,-1)
    mean_db=10*np.log10(np.maximum(bpower.mean(1),1e-30))
    ofloor=float(np.median(np.percentile(mean_db[:,~bad],10,axis=0)))
    occ=(10*np.log10(np.maximum(bpower.max(1),1e-30))>ofloor+10).mean(0); occ[bad]=0
    return db,f,floor,occ,n/fs,lo,hi

def detect(db,f,floor,dt,k):
    mask=db>floor+k
    mask=ndi.binary_closing(mask,structure=np.ones((5,3),bool))
    mask=ndi.binary_opening(mask,structure=np.ones((2,2),bool))
    # Fixed-tone columns remain excluded even if closing fills them.
    mask[:,np.all(db==floor,axis=0)]=False
    lab,n=ndi.label(mask,structure=ndi.generate_binary_structure(2,1))
    areas=np.bincount(lab.ravel()); ids=np.flatnonzero(areas>=20); ids=ids[ids!=0]
    if not len(ids): return []
    excess=db-floor
    sums=np.bincount(lab.ravel(),weights=excess.ravel(),minlength=len(areas))
    maxima=np.full(len(areas),-np.inf,dtype=np.float32)
    np.maximum.at(maxima,lab.ravel(),excess.ravel())
    means=sums[ids]/areas[ids]; peaks=maxima[ids]
    slices=ndi.find_objects(lab); df=f[1]-f[0]; rows=[]
    for idx,mean,mx in zip(ids,means,peaks):
        t,fr=slices[idx-1]
        rows.append(dict(box_id=len(rows)+1,t_start_ms=t.start*dt*1000,t_end_ms=t.stop*dt*1000,
                         f_start_mhz=float(f[fr.start]-df/2),f_end_mhz=float(f[fr.stop-1]+df/2),
                         area_px=int(areas[idx]),mean_excess_db=float(mean),max_excess_db=float(mx),
                         t0=t.start,t1=t.stop,f0=fr.start,f1=fr.stop))
    return rows

def merge(boxes,g,nt):
    ordered=sorted(boxes,key=lambda b:b['f_start_mhz']); groups=[]
    for b in ordered:
        if not groups or b['f_start_mhz']-groups[-1]['hi']>=g:
            groups.append(dict(lo=b['f_start_mhz'],hi=b['f_end_mhz'],boxes=[b]))
        else:
            groups[-1]['hi']=max(groups[-1]['hi'],b['f_end_mhz']); groups[-1]['boxes'].append(b)
    bands=[]
    for i,h in enumerate(groups,1):
        active=np.zeros(nt,bool)
        for b in h['boxes']: active[b['t0']:b['t1']]=True
        bands.append(dict(band_id=i,f_start_mhz=h['lo'],f_end_mhz=h['hi'],width_mhz=h['hi']-h['lo'],
                          time_duty=float(active.mean()),box_count=len(h['boxes']),max_excess_db=max(b['max_excess_db'] for b in h['boxes'])))
    return bands

def tag(k,g): return f'k{k}_g{g:g}'

def scene_for(c):
    p=c['parts'][0]['path']
    return '07-11家里' if '2026-07-11_home' in p else ('07-10实验室' if '2026-07-10_lab' in p else ('07-09城区' if c['source']=='raw' else '实验室'))

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    start=time.time(); before=snapshot(); dump('protected_before.json',before)
    clips=inventory()
    for c in clips: c['scene']=scene_for(c)
    dump('clips.json',clips)
    print('inventory',len(clips),collections.Counter((c['source'],c['label']) for c in clips),flush=True)
    # Only 07-09 background, same receiver center, is used in reference.
    refs=collections.defaultdict(list)
    for i,c in enumerate(clips):
        if c['source']=='raw' and c['label']=='ambient' and c['date']=='2026-07-09':
            db,f,fl,o,dt,lo,hi=spectrum(c); refs[c['fc']].append(o); check_resources()
    ref={fc:np.percentile(v,90,axis=0) for fc,v in refs.items()}
    dump('background_reference.json',{str(fc):{'clips':len(refs[fc]),'occupancy_p90':v.tolist()} for fc,v in ref.items()})
    allrows={tag(k,g):[] for k,g in PARAMS}; boxfiles={}; bandfiles={}; handles=[]
    boxkeys=['clip_id','box_id','t_start_ms','t_end_ms','f_start_mhz','f_end_mhz','area_px','mean_excess_db','max_excess_db']
    bandkeys=['clip_id','band_id','f_start_mhz','f_end_mhz','width_mhz','time_duty','box_count','max_excess_db']
    for k,g in PARAMS:
        t=tag(k,g)
        for prefix,keys,store in [('boxes',boxkeys,boxfiles),('bands',bandkeys,bandfiles)]:
            fh=(OUT/f'{prefix}_{t}.csv').open('w',encoding='utf-8-sig',newline=''); handles.append(fh)
            store[t]=csv.DictWriter(fh,fieldnames=keys,extrasaction='ignore'); store[t].writeheader()
    selected=collections.defaultdict(list); seen=set(); bgseen=set()
    # Earliest laboratory recording per model/band; one segment from that recording.
    labfirst={}
    for c in clips:
        if c['source']=='data':
            key=(c['model'],c['band']); labfirst[key]=min(labfirst.get(key,c['group']),c['group'],key=lambda s:s.split('__')[-1])
    for i,c in enumerate(clips):
        db,f,fl,o,dt,lo,hi=spectrum(c)
        if c['source']=='data':
            weights=np.where(db>fl+10,np.maximum(10.**((db-fl)/10)-1,0),0).sum(0,dtype=np.float64)
        elif c['label']=='drone':
            assert c['fc'] in ref, ('no same-center background',c['fc'])
            weights=np.maximum(o-ref[c['fc']],0)
        else: weights=np.zeros(len(f))
        total=float(weights.sum()); meta={x:c[x] for x in ['clip_id','source','band','model','state','bw','label','group','date','fc','start']}
        for k in (6,10):
            boxes=detect(db,f,fl,dt,k)
            for g in (.5,2):
                t=tag(k,g); bands=merge(boxes,g,len(db)); covered=np.zeros(len(f),bool)
                for b in bands: covered |= (f>=b['f_start_mhz'])&(f<b['f_end_mhz'])
                coverage=float(weights[covered].sum()/total) if total>0 else float('nan')
                row=dict(meta,coverage=coverage,reference_total=total,band_count=len(bands),box_count=len(boxes),floor_db=fl,
                         f_min_mhz=lo,f_max_mhz=hi,frames=len(db),bins=len(f))
                allrows[t].append(row)
                boxfiles[t].writerows(dict(b,clip_id=c['clip_id']) for b in boxes)
                bandfiles[t].writerows(dict(b,clip_id=c['clip_id']) for b in bands)
        # Fixed examples cached as float32: color is dB above this segment's floor.
        cat=None; key=(c['source'],c['model'],c['band'])
        if c['label']=='drone' and key not in seen and (c['source']=='raw' or c['group']==labfirst[(c['model'],c['band'])]):
            seen.add(key); cat=c['source']+'_drone'
        elif c['source']=='raw' and c['label']=='ambient' and (c['scene'],c['band']) not in bgseen:
            # One file per band from each scene folder; recording-clock date is separate.
            if c['slot']==0: bgseen.add((c['scene'],c['band'])); cat='raw_background'
        if cat:
            selected[cat].append(c['clip_id']); np.savez(CACHE/(c['clip_id']+'.npz'),db=db,f=f,floor=fl,dt=dt)
        if i%20==0:
            check_resources(); print(f'processed {i+1}/{len(clips)} elapsed={time.time()-start:.1f}s peak={peak/1024**3:.3f}GiB',flush=True)
    for h in handles: h.close()
    dump('selected_examples.json',dict(selected))
    for t,rows in allrows.items(): pd.DataFrame(rows).to_csv(OUT/f'segments_{t}.csv',index=False,encoding='utf-8-sig')
    dump('run_metrics.json',{'clips':len(clips),'peak_bytes':peak,'seconds_detection':time.time()-start,'background_reference_counts':{str(fc):len(v) for fc,v in refs.items()}})
    finish(clips,allrows,selected)

def finish(clips,allrows,selected):
    os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    byid={c['clip_id']:c for c in clips}; summary=[]; plotindex=[]
    quant=[0,.1,.25,.5,.75,.9,1]
    for k,g in PARAMS:
        t=tag(k,g); seg=pd.DataFrame(allrows[t]); bands=pd.read_csv(OUT/f'bands_{t}.csv'); boxes=pd.read_csv(OUT/f'boxes_{t}.csv')
        stats=[]
        def add(metric,values,**group):
            v=np.asarray(values,dtype=float); v=v[np.isfinite(v)]
            stats.append(dict(metric=metric,**group,n=len(v),**{f'q{int(q*100):02d}':float(np.quantile(v,q)) if len(v) else np.nan for q in quant}))
        for keys,ss in seg[seg.label=='drone'].groupby(['source','band','model','state','bw']):
            add('coverage',ss.coverage,**dict(zip(['source','band','model','state','bw'],keys)))
        for keys,ss in seg.groupby(['source','band','label']):
            for m in ('band_count','box_count'): add(m,ss[m],**dict(zip(['source','band','label'],keys)))
        merged=bands.merge(seg[['clip_id','source','band','label']],on='clip_id',validate='many_to_one')
        for keys,bb in merged.groupby(['source','band','label']):
            group=dict(zip(['source','band','label'],keys))
            add('boxes_per_band',bb.box_count,**group); add('band_width_mhz',bb.width_mhz,**group)
            stats.append(dict(metric='width_lt_1_fraction',**group,n=len(bb),value=float((bb.width_mhz<1).mean())))
        bg=merged[(merged.source=='raw')&(merged.label=='ambient')]
        near=np.zeros(len(bg),bool)
        for center in (2412,2437,2462): near|=(bg.f_start_mhz.to_numpy()<center+11)&(bg.f_end_mhz.to_numpy()>center-11)
        stats.append(dict(metric='background_wifi_overlap_fraction_all',n=len(bg),value=float(near.mean())))
        b24=bg[bg.band=='2.4GHz']; near24=np.zeros(len(b24),bool)
        for center in (2412,2437,2462): near24|=(b24.f_start_mhz.to_numpy()<center+11)&(b24.f_end_mhz.to_numpy()>center-11)
        stats.append(dict(metric='background_wifi_overlap_fraction_2p4',n=len(b24),value=float(near24.mean())))
        pd.DataFrame(stats).to_csv(OUT/f'stats_{t}.csv',index=False,encoding='utf-8-sig')
        s=dict(parameter=t,boxes=len(boxes),bands=len(bands),bands_median=float(seg.band_count.median()),bands_p90=float(seg.band_count.quantile(.9)),
               narrow_fraction=float((bands.width_mhz<1).mean()),wifi_fraction=float(near24.mean()),zero_reference=int(((seg.label=='drone')&(seg.reference_total==0)).sum()))
        for src in ('data','raw'):
            v=seg[(seg.source==src)&(seg.label=='drone')].coverage.dropna(); s[src+'_median']=float(v.median());s[src+'_min']=float(v.min())
        summary.append(s)
        selections=dict(selected); worst=seg[seg.label=='drone'].sort_values(['coverage','clip_id']).head(4).clip_id.tolist(); selections['lowest_coverage']=worst
        for category,ids in selections.items():
            for page,offset in enumerate(range(0,len(ids),8),1):
                groupids=ids[offset:offset+8]; fig,axes=plt.subplots(len(groupids),2,figsize=(19,3.1*len(groupids)),squeeze=False)
                for row,cid in enumerate(groupids):
                    c=byid[cid]; cp=CACHE/(cid+'.npz')
                    if cp.exists():
                        with np.load(cp) as z: db=z['db'];f=z['f'];floor=float(z['floor']);dt=float(z['dt'])
                    else:
                        db,f,floor,o,dt,lo,hi=spectrum(c); np.savez(cp,db=db,f=f,floor=floor,dt=dt)
                    df=f[1]-f[0]; extent=[f[0]-df/2,f[-1]+df/2,0,len(db)*dt*1000]
                    ss=seg[seg.clip_id==cid].iloc[0]; bs=boxes[boxes.clip_id==cid]; bb=bands[bands.clip_id==cid]
                    for col,ax in enumerate(axes[row]):
                        im=ax.imshow((db-floor),origin='lower',aspect='auto',extent=extent,vmin=-5,vmax=35,cmap='viridis',rasterized=True)
                        ax.set_xlabel('频率（MHz）');ax.set_ylabel('片段时间（ms）')
                        title=f'{cid} '+('实验室' if c['source']=='data' else c.get('scene','城区'))+f" {c['model']} {c['band']} {c['state']} 时钟{c['date']}"
                        ax.set_title(title+('｜全频段' if col==0 else f'｜k={k} dB，g={g:g} MHz，{len(bb)} 带 / {len(bs)} 框'),fontsize=9)
                        if col:
                            for _,b in bb.iterrows():
                                ax.axvspan(b.f_start_mhz,b.f_end_mhz,color='orange',alpha=.18)
                                ax.text((b.f_start_mhz+b.f_end_mhz)/2,extent[3]*.98,f'{int(b.band_id)}: {b.f_start_mhz:.2f}–{b.f_end_mhz:.2f}',rotation=90,va='top',ha='center',fontsize=6,color='darkred',bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=.1))
                            if len(bs):
                                x0=bs.f_start_mhz.to_numpy();x1=bs.f_end_mhz.to_numpy();y0=bs.t_start_ms.to_numpy();y1=bs.t_end_ms.to_numpy()
                                vertices=np.stack([np.stack([x0,y0],1),np.stack([x1,y0],1),np.stack([x1,y1],1),np.stack([x0,y1],1),np.stack([x0,y0],1)],1)
                                ax.add_collection(LineCollection(vertices,colors='white',linewidths=.35))
                    cov='未定义' if not np.isfinite(ss.coverage) else f'{ss.coverage*100:.2f}%'
                    axes[row,0].text(.01,.02,f'覆盖率 {cov}；起点 {c["start"]/c["fs"]:.6f} s；底噪 {floor:.2f} dB',transform=axes[row,0].transAxes,color='white',fontsize=8)
                    plotindex.append(dict(parameter=t,category=category,page=page,row=row+1,clip_id=cid,coverage=ss.coverage))
                fig.subplots_adjust(left=.055,right=.94,bottom=.04,top=.97,hspace=.42,wspace=.15)
                cb=fig.colorbar(im,cax=fig.add_axes([.955,.15,.008,.7]));cb.set_label('相对本片段底噪（dB），统一 −5～35')
                fig.savefig(OUT/f'图_{t}_{category}_{page:02d}.png',dpi=120);plt.close(fig);check_resources()
        print('figures finished',t,flush=True)
    pd.DataFrame(plotindex).to_csv(OUT/'plot_index.csv',index=False,encoding='utf-8-sig')
    dump('summary.json',summary)
    after=snapshot();dump('protected_after.json',after)
    verify={'protected_identical':after==json.loads((OUT/'protected_before.json').read_text(encoding='utf-8')),
            'protected_files':len(after),'peak_bytes':max(peak,PROC.memory_info().peak_wset),
            'cache_bytes':sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()),
            'figures':len(list(OUT.glob('图_*.png'))),'plot_rows':len(plotindex),'all_clips_per_parameter':{t:len(v) for t,v in allrows.items()}}
    assert verify['protected_identical']; assert verify['peak_bytes']<8*1024**3; dump('verification.json',verify)
    report(clips,summary,verify)

def report(clips,summary,v):
    met=json.loads((OUT/'run_metrics.json').read_text(encoding='utf-8'))
    lines=['# 自动找信号频带试验 Q（2026-11-11）','',
    f'处理片段 {len(clips)}；实验室录制 {len(set(c["group"] for c in clips if c["source"]=="data"))}，片段 {sum(c["source"]=="data" for c in clips)}；城区文件 {len(set(c["parts"][0]["path"] for c in clips if c["source"]=="raw"))}，片段 {sum(c["source"]=="raw" for c in clips)}。',
    f'城区无人机片段 {sum(c["source"]=="raw" and c["label"]=="drone" for c in clips)}，城区背景片段 {sum(c["label"]=="ambient" for c in clips)}；每段 20 ms。',
    '实验室每次录制在有效起点区间的并集内均匀取 6 段，不跨无效区间；同一录制 seg/epoch 首尾拼接。',
    '城区每个 0.12 s 文件取 [0,20) ms 和 [50,70) ms。未使用重采样数据。',
    '分析范围：2.4G 为 [max(2395,中心−fs/2+1.5),min(2488.5,中心+fs/2−1.5)] MHz；5.8G 同式限于 5720–5855 MHz。',
    '实验室 2.4G 分析上限 2459.94 MHz，实际最后频点 2459.88 MHz；2461 MHz 以上不在录音里。',
    'Hann 窗、不重叠：实验室 nfft/hop=1024，120 kHz、8.333 µs；城区 2048，122.070 kHz、8.192 µs。',
    '已知谱线 ±0.3 MHz 排除底噪估计后置为 F；F 为其余全部像素 dB 第 20 百分位。形态学后再次清零谱线列。',
    '闭运算 时间5×频率3，再开运算 时间2×频率2；四邻域；保留面积≥20像素。框使用连通域像素的均值与峰值。',
    '频带以框的频率区间按间隙严格小于 g 合并；时间占空比为所属框时间区间并集/全部帧；所有区间左闭右开。',
    '实验室覆盖率参照：高于 F+10 dB 像素的线性超底噪能量；频带覆盖按整个频率区间计算。',
    '城区参照沿用 outside_crop_v3.py：1 ms 内最大功率占用率；其门限底噪为逐频率时间均功率第10百分位再取中位，门限+10 dB。',
    '城区无人机占用率减同中心 07-09 背景逐频率第90百分位，负数置0；分辨率改为本任务要求，去除谱线，仅在分析范围内计算。',
    f'城区背景参考片段数（中心 MHz）：{met["background_reference_counts"]}；参照总量为0时覆盖率写空值，最低覆盖率图不选这些片段。','',
    '|参数|框数|频带数|每片段频带数 中位/P90|实验室覆盖率 中位/最小|城区覆盖率 中位/最小|<1MHz频带占比|城区2.4G背景WiFi重叠占比|零参照片段|',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f'|{s["parameter"]}|{s["boxes"]}|{s["bands"]}|{s["bands_median"]:.1f}/{s["bands_p90"]:.1f}|{s["data_median"]:.2%}/{s["data_min"]:.2%}|{s["raw_median"]:.2%}/{s["raw_min"]:.2%}|{s["narrow_fraction"]:.2%}|{s["wifi_fraction"]:.2%}|{s["zero_reference"]}|')
    lines += ['',
    'WiFi 附近按频带与 2412/2437/2462±11 MHz 任一区间重叠计；stats 表另列全部背景频带为分母的比例。',
    'stats_*.csv：按来源/频段/机型/状态/带宽列覆盖率 q00/q50，按来源/频段/有无无人机列频带数、框数分位数；另列背景宽度、框/频带、窄带占比。',
    'segments_*.csv 含全部片段的覆盖率、参照总量、框数、频带数；boxes_*.csv、bands_*.csv 使用 clip_id 关联 clips.json。',
    '实验室标签仅复用已有 plan.json 的描述字段；取样完全由 data_usable_runs.json 的区间决定。城区未知状态/带宽保持 unknown。',
    '每组图包括实验室每机型/频段最早录制1段、城区每机型/频段1段、07-09城区/07-10实验室/07-11家里背景各2个不同文件，以及该参数覆盖率最低4段。',
    '背景场景按文件夹区分；2026-07-11_home 内 session.id 为 20260710-11xxxx，文件夹是场景标签，日期是录音时钟日期，两者不一致。',
    f'输出图 {v["figures"]} 张，每张≤8行；plot_index.csv 共 {v["plot_rows"]} 行；左右两图颜色统一为相对片段底噪 −5～35 dB，频带橙色、框白线。',
    f'保护目录文件 {v["protected_files"]} 个，前后 SHA256 完全一致={v["protected_identical"]}；详见 protected_before/after.json。',
    f'进程峰值内存 {v["peak_bytes"]/1024**3:.3f} GiB；E盘缓存 {v["cache_bytes"]/10**9:.3f} GB；本次恢复处理 {met["seconds_detection"]:.1f} s。',
    f'复用暂停前完整框和频带 {met.get("legacy_reused_clips",0)} 段并补算覆盖率；暂停前最后日志累计 {met.get("seconds_previous_run_last_logged",0):.1f} s。',
    '运行：KMP_DUPLICATE_LIB_OK=TRUE、PYTHONIOENCODING=utf-8；D:/anaconda/python.exe -I work/side_Q_band_detect_pilot_20261111/pilot.py --resume。',
    '仅逐片段读取 G盘 data/raw；MAT 每次仅载入必要的单个50ms容器；未读取F盘；未训练、未装包、未提交。']
    assert len(lines)<=80
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def resume_main():
    """Recover legacy partial outputs, then commit every complete clip durably."""
    start=time.time(); current=snapshot(); original=json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'))
    assert current==original, 'protected directory changed since original baseline'
    clips=json.loads((OUT/'clips.json').read_text(encoding='utf-8'))
    for c in clips: c['scene']=scene_for(c)
    dump('clips.json',clips)
    refs=json.loads((OUT/'background_reference.json').read_text(encoding='utf-8'))
    ref={float(fc):np.array(v['occupancy_p90']) for fc,v in refs.items()}
    selected=collections.defaultdict(list); seen=set()
    for c in sorted(clips,key=lambda c:(c['group'].split('__')[-1],c['slot'])):
        key=(c['source'],c['model'],c['band'])
        if c['label']=='drone' and key not in seen:
            seen.add(key);selected[c['source']+'_drone'].append(c['clip_id'])
    for scene in ('07-09城区','07-10实验室','07-11家里'):
        eligible=[c for c in clips if c['source']=='raw' and c['label']=='ambient' and c['scene']==scene and c['slot']==0]
        assert len(eligible)>=2
        # Prefer one example per band, using distinct files.
        chosen=[]
        for band in ('2.4GHz','5.8GHz'):
            a=next((c for c in eligible if c['band']==band),None)
            if a: chosen.append(a)
        for c in eligible:
            if len(chosen)>=2: break
            if c not in chosen: chosen.append(c)
        selected['raw_background'].extend(c['clip_id'] for c in chosen)
    dump('selected_examples.json',dict(selected)); wanted={x for v in selected.values() for x in v}
    boxkeys=['clip_id','box_id','t_start_ms','t_end_ms','f_start_mhz','f_end_mhz','area_px','mean_excess_db','max_excess_db']
    bandkeys=['clip_id','band_id','f_start_mhz','f_end_mhz','width_mhz','time_duty','box_count','max_excess_db']
    metakeys=['clip_id','source','band','model','state','bw','label','group','date','fc','start']
    segkeys=metakeys+['coverage','reference_total','band_count','box_count','floor_db','f_min_mhz','f_max_mhz','frames','bins']
    statepath=OUT/'resume_state.json'
    if statepath.exists():
        state=json.loads(statepath.read_text(encoding='utf-8'))
        for name,size in state['positions'].items():
            with (OUT/name).open('r+b') as fh: fh.truncate(size)
    else:
        # Exclude the last observed clip in every legacy stream: its rows may be partial.
        ends=[]
        for prefix in ('bands','boxes'):
            for k,g in PARAMS:
                last=None
                for chunk in pd.read_csv(OUT/f'{prefix}_{tag(k,g)}.csv',usecols=['clip_id'],chunksize=200000): last=chunk.clip_id.iloc[-1]
                ends.append(int(last[1:]))
        cut=min(ends);positions={}
        for prefix in ('bands','boxes'):
            for k,g in PARAMS:
                path=OUT/f'{prefix}_{tag(k,g)}.csv'; tmp=path.with_suffix('.recovering.csv');first=True
                for chunk in pd.read_csv(path,chunksize=100000):
                    chunk=chunk[chunk.clip_id < f'Q{cut:04d}']
                    chunk.to_csv(tmp,mode='w' if first else 'a',header=first,index=False,encoding='utf-8-sig' if first else 'utf-8');first=False
                tmp.replace(path);positions[path.name]=path.stat().st_size
        for k,g in PARAMS:
            path=OUT/f'segments_{tag(k,g)}.csv'
            with path.open('w',encoding='utf-8-sig',newline='') as fh: csv.DictWriter(fh,fieldnames=segkeys).writeheader()
            positions[path.name]=path.stat().st_size
        state=dict(next_clip=0,reuse_before=cut,positions=positions,seconds_resume=0)
        dump('resume_state.json',state)
    reused={}
    for k,g in PARAMS:
        d=pd.read_csv(OUT/f'bands_{tag(k,g)}.csv');reused[tag(k,g)]={cid:v.to_dict('records') for cid,v in d.groupby('clip_id')}
    handles={};writers={}
    for k,g in PARAMS:
        t=tag(k,g)
        for prefix,keys in [('boxes',boxkeys),('bands',bandkeys),('segments',segkeys)]:
            name=f'{prefix}_{t}.csv';fh=(OUT/name).open('a',encoding='utf-8',newline='');handles[name]=fh
            writers[name]=csv.DictWriter(fh,fieldnames=keys,extrasaction='ignore')
    print('resume',state['next_clip'],'reuse legacy before',state['reuse_before'],flush=True)
    previous_seconds=state.get('seconds_resume',0)
    for i in range(state['next_clip'],len(clips)):
        c=clips[i];db,f,fl,o,dt,lo,hi=spectrum(c)
        weights=(np.where(db>fl+10,np.maximum(10.**((db-fl)/10)-1,0),0).sum(0,dtype=np.float64) if c['source']=='data' else
                 (np.maximum(o-ref[c['fc']],0) if c['label']=='drone' else np.zeros(len(f))))
        total=float(weights.sum());meta={key:c[key] for key in metakeys}
        for k in (6,10):
            boxes=detect(db,f,fl,dt,k) if i>=state['reuse_before'] else None
            for g in (.5,2):
                t=tag(k,g)
                if boxes is None: bands=reused[t].get(c['clip_id'],[])
                else:
                    bands=merge(boxes,g,len(db));writers[f'boxes_{t}.csv'].writerows(dict(b,clip_id=c['clip_id']) for b in boxes)
                    writers[f'bands_{t}.csv'].writerows(dict(b,clip_id=c['clip_id']) for b in bands)
                covered=np.zeros(len(f),bool)
                for b in bands: covered|=(f>=b['f_start_mhz'])&(f<b['f_end_mhz'])
                writers[f'segments_{t}.csv'].writerow(dict(meta,coverage=float(weights[covered].sum()/total) if total>0 else float('nan'),
                    reference_total=total,band_count=len(bands),box_count=sum(b['box_count'] for b in bands),floor_db=fl,
                    f_min_mhz=lo,f_max_mhz=hi,frames=len(db),bins=len(f)))
        if c['clip_id'] in wanted and not (CACHE/(c['clip_id']+'.npz')).exists():
            np.savez(CACHE/(c['clip_id']+'.npz'),db=db,f=f,floor=fl,dt=dt)
        for fh in handles.values(): fh.flush()
        state.update(next_clip=i+1,positions={name:fh.tell() for name,fh in handles.items()},seconds_resume=previous_seconds+time.time()-start)
        dump('resume_state.json',state)
        if i%20==0:
            check_resources();print(f'committed {i+1}/{len(clips)} reuse={i<state["reuse_before"]} elapsed={time.time()-start:.1f}s peak={peak/1024**3:.3f}GiB',flush=True)
    for fh in handles.values(): fh.close()
    dump('run_metrics.json',{'clips':len(clips),'peak_bytes':peak,'seconds_detection':state['seconds_resume'],
        'seconds_previous_run_last_logged':1936.3,'legacy_reused_clips':state['reuse_before'],
        'background_reference_counts':{fc:v['clips'] for fc,v in refs.items()}})
    rows={tag(k,g):pd.read_csv(OUT/f'segments_{tag(k,g)}.csv').to_dict('records') for k,g in PARAMS}
    finish(clips,rows,selected)

if __name__=='__main__':
    if '--finish' in sys.argv:
        clips=json.loads((OUT/'clips.json').read_text(encoding='utf-8'))
        rows={tag(k,g):pd.read_csv(OUT/f'segments_{tag(k,g)}.csv').to_dict('records') for k,g in PARAMS}
        finish(clips,rows,json.loads((OUT/'selected_examples.json').read_text(encoding='utf-8')))
    elif '--resume' in sys.argv: resume_main()
    else: main()
