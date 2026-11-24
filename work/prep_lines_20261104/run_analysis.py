import os,sys,json,csv,re,collections,time
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import numpy as np
from scipy import signal,ndimage,io
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lab_preproc import *
OUT=Path(__file__).resolve().parent; ROOT=Path('G:/drone'); PROJECT=OUT.parent.parent
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def csvout(name,rows,fields=None):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def snapshot():
    n=0; latest=0
    for base,dirs,files in os.walk(ROOT):
        for name in files:
            s=(Path(base)/name).stat(); n+=1; latest=max(latest,s.st_mtime_ns)
    return dict(count=n,latest_mtime_ns=latest)
def psd(x,fs):
    f,p=signal.welch(x,fs,nperseg=65536,noverlap=32768,window='hann',return_onesided=False)
    return np.fft.fftshift(f),np.fft.fftshift(p)
def detect(p,fs):
    db=10*np.log10(p+1e-30); step=fs/65536
    floor=ndimage.median_filter(db,size=(int(2e6/step)|1),mode='nearest'); peaks,_=signal.find_peaks(db-floor,height=10,distance=max(1,int(20000/step)))
    found=[]
    for k in peaks:
        l=r=k
        while l>0 and db[l]>db[k]-10:l-=1
        while r<len(db)-1 and db[r]>db[k]-10:r+=1
        width=(r-l)*step
        if width<=50000:found.append(dict(k=int(k),width_khz=width/1000,level_db_over_local=float(db[k]-floor[k])))
    return found
def occupancy(chunks,fs,fc):
    cells=[]; size=round(fs*.001); win=np.hanning(4096); norm=sum(win**2)
    for x in chunks:
        for a in range(0,len(x)-size+1,size):
            frames=x[a:a+size][:size//4096*4096].reshape(-1,4096)
            power=abs(np.fft.fftshift(np.fft.fft(frames*win,axis=1),axes=1))**2/norm
            power=power.reshape(-1,1024,4).mean(2)
            cells.append(10*np.log10(power.max(0)+1e-20))
    cells=np.array(cells); freq=fc+np.mean(np.fft.fftshift(np.fft.fftfreq(4096,1/fs)).reshape(1024,4),axis=1)/1e6
    mask=abs(freq-fc)<=.45*fs/1e6; mask[511:514]=False
    floor=np.median(np.percentile(cells[:,mask],10,axis=0)); return freq,np.mean(cells>floor+10,axis=0),mask
def measure(rec,chunks):
    fs=rec['fs']; fc=rec['fc']; spectra=[psd(x,fs)[1] for x in chunks]; p=np.mean(spectra,axis=0); f=np.fft.fftshift(np.fft.fftfreq(65536,1/fs))/1e6+fc
    lines=detect(p,fs); blocks=[x[a:a+round(fs*.01)] for x in chunks for a in range(0,len(x)-round(fs*.01)+1,round(fs*.01))]
    bps=np.array([psd(x,fs)[1] for x in blocks]); bdb=10*np.log10(bps+1e-30)
    for ln in lines:
        k=ln.pop('k'); near=np.where(abs(f-f[k])<=.02)[0]; local=np.where((abs(f-f[k])<=1)&(abs(f-f[k])>.15))[0]
        idx=near[np.argmax(bps[:,near],axis=1)]; level=bdb[np.arange(len(idx)),idx]-np.median(bdb[:,local],axis=1)
        present=level>=10
        ln.update(freq_mhz=float(f[k]),offset_from_fc_khz=(f[k]-fc)*1000,presence_rate=float(np.mean(present)),drift_khz=float(np.ptp(f[idx[present]])*1000) if present.any() else 0,amp_std_db=float(np.std(bdb[np.arange(len(idx)),idx])),is_at_fc=bool(abs(f[k]-fc)<=.3))
    if rec['dataset']=='data':
        cache=Path('E:/drone_lab_cache_20261031')/rec['key']; occ=np.load(cache/'occ_any.npy'); cfg=json.loads((PROJECT/'work/lab_long_overview_20261031/config.json').read_text(encoding='utf-8')); freq=np.array(cfg['freq_mhz_centers'][rec['band']]); mask=np.array(cfg['mask'],bool)
    else:freq,occ,mask=occupancy(chunks,fs,fc)
    clean=occ.copy()
    for ln in lines:clean[abs(freq-ln['freq_mhz'])<=.15]=0
    for ln in lines:
        # Line bins are excluded; nearest surrounding occupancy estimates underlying signal.
        region=(abs(freq-ln['freq_mhz'])<=.30)&(abs(freq-ln['freq_mhz'])>.15)&mask
        ln['overlaps_signal']=bool(rec['label']=='drone' and np.any(occ[region]>=.05))
        ln.update(group=rec['group'],recording=rec['key'])
    centers=freq[(freq-28>=fc-.45*fs/1e6)&(freq+28<=fc+.45*fs/1e6)]
    weights=np.array([clean[mask&(abs(freq-c)<=28)].sum() for c in centers]); center=float(centers[np.argmax(weights)]) if len(centers) else fc
    ratio=float(max(weights)/clean[mask].sum()) if clean[mask].sum()>0 else None
    rec.update(center_mhz=center,occupancy_fraction=ratio,measured_ms=sum(map(len,chunks))/fs*1000,n_lines=len(lines))
    validations=[]
    for ln in lines:
        off=(ln['freq_mhz']-fc)*1e6; region=(abs(f-ln['freq_mhz'])>=.15)&(abs(f-ln['freq_mhz'])<=2)
        for bi,x in enumerate(blocks[:3]):
            for method in ['cancel','notch']+(['mean'] if abs(off)<fs/65536 else []):
                y=notch(x,fs,off,ln['width_khz']*1000) if method=='notch' else cancel_tone(x,fs,off,mean_only=method=='mean')
                before=psd(x,fs)[1]; after=psd(y,fs)[1]; local=(abs(f-ln['freq_mhz'])<=1)&(abs(f-ln['freq_mhz'])>.15); core=abs(f-ln['freq_mhz'])<=.025
                residual=float(10*np.log10(after[core].max()/np.median(after[local]))); delta=float(10*np.log10(after[region].sum()/before[region].sum()))
                validations.append(dict(group=rec['group'],recording=rec['key'],freq_mhz=ln['freq_mhz'],overlaps_signal=ln['overlaps_signal'],block=bi,method=method,block_ms=10,residual_db=residual,off_line_delta_db=delta,error_energy_ratio=float(np.mean(abs(y-x)**2)/np.mean(abs(x)**2)),passed=residual<=3 and abs(delta)<=.1))
    # Individual spectra also preserve evidence before aggregation.
    np.savez_compressed(OUT/('spectrum_'+re.sub(r'[^\w.-]','_',rec['key'])+'.npz'),freq=f,power=p)
    return lines,validations,f,p

def process_record(rec):
    key=rec['key']; checkpoint=OUT/('checkpoint_'+re.sub(r'[^\w.-]','_',key)+'.json')
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding='utf-8'));rec.update(saved['record']);spec=np.load(OUT/('spectrum_'+re.sub(r'[^\w.-]','_',key)+'.npz'))
        return rec,saved['lines'],saved['validations'],spec['freq'],spec['power']
    files=sorted(rec['files'],key=lambda m:m['path']); selected=[files[i] for i in np.unique(np.linspace(0,len(files)-1,min(5,len(files))).astype(int))];chunks=[]
    for m in selected:
        var='data' if rec['dataset']=='data' else 'IQ';x=io.loadmat(ROOT/m['path'],variable_names=[var])[var].ravel()
        if rec['dataset']=='data':x=x[int(m['start_sample']):int(m['end_sample_exclusive'])]
        chunks.append(x)
    lines,vals,f,p=measure(rec,chunks)
    dump(checkpoint.name,dict(record={k:v for k,v in rec.items() if k!='files'},lines=lines,validations=vals))
    return rec,lines,vals,f,p

def main():
    before=json.loads((OUT/'raw_snapshot_before.json').read_text(encoding='utf-8')) if (OUT/'raw_snapshot_before.json').exists() else snapshot(); dump('raw_snapshot_before.json',before)
    manifest=list(csv.DictReader((PROJECT/'work/audit_drone_20261023/data_preprocessing_manifest.csv').open(encoding='utf-8-sig')))
    meta=json.loads((PROJECT/'work/verify_v3_20261104/v3_meta.json').read_text(encoding='utf-8'))
    sources=[]; records={}; all_lines=[]; validations=[]; specs={}
    for m in manifest:
        key=m['folder']+'__'+m['session']; band='2.4GHz' if '2.4' in m['folder'] else '5.8GHz'; duplicate=m['duplicate']=='True'
        sources.append(dict(dataset='data',path=m['path'].replace('\\','/'),label='drone',model=m['folder'].split('_')[0],band=band,group_id='data:'+key,include=not duplicate and m['analysis_eligible']=='True',exclude_reason='duplicate' if duplicate else m['exclusion_reason']))
        if not duplicate and m['analysis_eligible']=='True':records.setdefault(key,dict(key=key,group='data_'+band,dataset='data',label='drone',band=band,fc=2400 if band=='2.4GHz' else 5800,fs=122880000,files=[]))['files'].append(m)
    vrecords={}
    for m in meta:
        name=m['name']; fc=m['fc_name']/1e6; match=re.search(r'(\d{8}-\d{6}_[a-zA-Z0-9]+)_iq',name); session=match.group(1) if match else name.split('_iq')[0]; scan=name.startswith('2026-07-16_lab_'); key=session+('@'+str(fc) if scan else '')
        label='background' if m['sub']=='background' else 'drone'; allowed=label=='drone' or name.startswith(('2026-07-09_urban_','2026-07-10_lab_','2026-07-11_home_','noise_adrv_2450_','noise_adrv_5788_')) or (scan and fc in [2367.06,2440,5760,5810,5881.56])
        path='datasets_v3/'+m['sub'].replace('\\','/')+'/'+name; band='2.4GHz' if fc<4000 else '5.8GHz'
        sources.append(dict(dataset='v3',path=path,label=label,model=m['sub'].split('\\')[-1],band=band,group_id='v3:'+key,include=allowed,exclude_reason='' if allowed else 'out_of_scope_frequency_or_session'))
        if allowed:vrecords.setdefault(key,dict(key=key,group='v3_'+key,dataset='v3',label=label,band=band,fc=fc,fs=100000000,files=[]))['files'].append(dict(path=path))
    csvout('source_groups.csv',sources)
    # Verify one selected scan against ci16 interleaved raw, using actual s1 boundary search.
    scanrec=next(r for r in vrecords.values() if '@' in r['key']); vp=ROOT/scanrec['files'][0]['path']; rawdir=ROOT/'raw/2026-07-09_urban/ambient/2026-07-16_lab/20260715-075302_ambient'
    rawname=vp.name.split('_ambient_iq_')[1].split('_s000')[0]; rawpath=rawdir/('iq_'+rawname+'.bin')
    raw=np.fromfile(rawpath,dtype='<i2').reshape(-1,2); z=raw[:,0].astype(float)+1j*raw[:,1]; z=signal.resample_poly(z,4,5); target=io.loadmat(vp,variable_names=['IQ'])['IQ'].ravel()
    corr=signal.correlate(z[:30000],target[:4096],mode='valid',method='fft'); offset=int(np.argmax(abs(corr))); a=z[offset:offset+len(target)]; correlation=float(abs(np.vdot(a,target))/np.linalg.norm(a)/np.linalg.norm(target)); scancheck=dict(path=str(vp),raw_path=str(rawpath),offset_samples=offset,correlation=correlation,passed=correlation>.999)
    dump('scan_verification.json',scancheck); del raw,z,target,a
    if not scancheck['passed']:
        for s in sources:
            if '@' in s['group_id']:s.update(include=False,exclude_reason='scan_raw_mismatch')
        vrecords={k:r for k,r in vrecords.items() if '@' not in k}; csvout('source_groups.csv',sources)
    records.update(vrecords)
    pool=ThreadPoolExecutor(max_workers=3)
    futures=[pool.submit(process_record,r) for r in records.values()]
    for ri,future in enumerate(as_completed(futures)):
        rec,lines,vals,f,p=future.result(); key=rec['key'];all_lines.extend(lines);validations.extend(vals)
        if rec['group'] not in specs:specs[rec['group']]=[f,p,1]
        else:specs[rec['group']][1]+=p;specs[rec['group']][2]+=1
        print(f'{ri+1}/{len(records)} {key} lines={len(lines)} ms={rec["measured_ms"]:.1f}',flush=True)
        dump('progress.json',dict(done=ri+1,total=len(records),key=key))
    pool.shutdown()
    # Sequential implementation retained below as an inactive reference.
    completed_records=records
    records={}
    for ri,(key,rec) in enumerate(records.items()):
        checkpoint=OUT/('checkpoint_'+re.sub(r'[^\w.-]','_',key)+'.json')
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text(encoding='utf-8'));rec.update(saved['record']);lines=saved['lines'];vals=saved['validations']; spectrum=np.load(OUT/('spectrum_'+re.sub(r'[^\w.-]','_',key)+'.npz'));f=spectrum['freq'];p=spectrum['power']
            all_lines.extend(lines);validations.extend(vals)
            if rec['group'] not in specs:specs[rec['group']]=[f,p,1]
            else:specs[rec['group']][1]+=p;specs[rec['group']][2]+=1
            print(f'{ri+1}/{len(records)} {key} checkpoint restored',flush=True);continue
        files=sorted(rec['files'],key=lambda m:m['path']); count=5 if rec['dataset']=='v3' else 5
        selected=[files[i] for i in np.unique(np.linspace(0,len(files)-1,min(count,len(files))).astype(int))]; chunks=[]
        for m in selected:
            var='data' if rec['dataset']=='data' else 'IQ'; x=io.loadmat(ROOT/m['path'],variable_names=[var])[var].ravel()
            if rec['dataset']=='data':x=x[int(m['start_sample']):int(m['end_sample_exclusive'])]
            chunks.append(x)
        lines,vals,f,p=measure(rec,chunks); all_lines.extend(lines); validations.extend(vals)
        dump(checkpoint.name,dict(record={k:v for k,v in rec.items() if k!='files'},lines=lines,validations=vals))
        if rec['group'] not in specs:specs[rec['group']]=[f,p,1]
        else:specs[rec['group']][1]+=p; specs[rec['group']][2]+=1
        print(f'{ri+1}/{len(records)} {key} lines={len(lines)} ms={rec["measured_ms"]:.1f}',flush=True)
        dump('progress.json',dict(done=ri+1,total=len(records),key=key)); del chunks,x
    records=completed_records
    csvout('lines_by_recording.csv',all_lines); csvout('cancel_validation.csv',validations)
    inventory=[]
    for group in specs:
        members=[r for r in records.values() if r['group']==group]; ls=sorted([l for l in all_lines if l['group']==group],key=lambda l:l['freq_mhz']); clusters=[]
        for ln in ls:
            if not clusters or ln['freq_mhz']-np.mean([v['freq_mhz'] for v in clusters[-1]])>.02:clusters.append([])
            clusters[-1].append(ln)
        for cluster in clusters:
            row={k:float(np.mean([l[k] for l in cluster])) for k in ['freq_mhz','offset_from_fc_khz','width_khz','level_db_over_local','presence_rate','drift_khz','amp_std_db']}; row.update(group=group,is_at_fc=all(l['is_at_fc'] for l in cluster),n_recordings_with_line=len(cluster),n_recordings=len(members)); inventory.append(row)
        f,p,n=specs[group]; lines=[l for l in inventory if l['group']==group]; fig,axes=plt.subplots(1+min(20,len(lines)),1,figsize=(13,max(4,2.3*(1+min(20,len(lines)))))); axes=np.atleast_1d(axes)
        for ax in axes:ax.plot(f,10*np.log10(p/n+1e-30),lw=.7);ax.set_xlabel('Absolute frequency (MHz)');ax.set_ylabel('PSD (dB/Hz)');ax.grid(alpha=.2)
        axes[0].set_title(group)
        for ln,ax in zip(lines,axes[1:]):ax.set_xlim(ln['freq_mhz']-2,ln['freq_mhz']+2);ax.axvline(ln['freq_mhz'],c='r',lw=.7);axes[0].axvline(ln['freq_mhz'],c='r',lw=.5)
        fig.tight_layout();fig.savefig(OUT/('lines_'+re.sub(r'[^\w.-]','_',group)+'.png'),dpi=140);plt.close(fig)
        for page in range(20,len(lines),20):
            part=lines[page:page+20]; fig,axes=plt.subplots(len(part),1,figsize=(13,max(4,2.3*len(part))));axes=np.atleast_1d(axes)
            for ln,ax in zip(part,axes):ax.plot(f,10*np.log10(p/n+1e-30),lw=.7);ax.set_xlim(ln['freq_mhz']-2,ln['freq_mhz']+2);ax.set_xlabel('Absolute frequency (MHz)');ax.axvline(ln['freq_mhz'],c='r')
            fig.tight_layout();fig.savefig(OUT/('lines_'+re.sub(r'[^\w.-]','_',group)+f'_page{page//20+1}.png'),dpi=140);plt.close(fig)
    csvout('lines_inventory.csv',inventory)
    plans=[]; backgrounds=[]
    for rec in records.values():
        if rec['label']=='drone':centers=[rec['center_mhz']]
        else:centers=sorted(set(r['center_mhz'] for r in records.values() if r['label']=='drone' and r['band']==rec['band'] and abs(r['center_mhz']-rec['fc'])<=17))
        ls=[l for l in all_lines if l['recording']==rec['key']]
        if rec['label']=='background':
            lo,hi=(2400,2483.5) if rec['band']=='2.4GHz' else (5725,5850); overlap=max(0,min(hi,rec['fc']+45)-max(lo,rec['fc']-45)); backgrounds.append(dict(session=rec['key'],usable_low_mhz=rec['fc']-45,usable_high_mhz=rec['fc']+45,drone_overlap_mhz=overlap,in_drone_band='False' if overlap==0 else 'partial',candidate_count=len(centers),centers_mhz=json.dumps(centers),status='ok' if centers else 'no_valid_center'))
        for center in centers or [None]:
            actions=[dict(freq_mhz=l['freq_mhz'],method=('cancel' if l['overlaps_signal'] else 'notch') if center is not None and abs(l['freq_mhz']-center)<=28 else 'removed_by_bandpass') for l in ls]
            plans.append(dict(recording=rec['key'],dataset=rec['dataset'],label=rec['label'],band=rec['band'],fc_mhz=rec['fc'],center_mhz=center,low_mhz=center-28 if center else None,high_mhz=center+28 if center else None,occupancy_fraction=rec['occupancy_fraction'] if rec['label']=='drone' else None,includes_fc=abs(center-rec['fc'])<=28 if center else False,lines=json.dumps(actions),status='ok' if center else 'no_valid_center'))
    csvout('crop_plan.csv',plans);csvout('background_sessions.csv',backgrounds)
    tests=[]
    for fs in [100000000,122880000]:
        for off in [-35e6,-20e6,20e6,35e6]:
            t=np.arange(round(fs*.003))/fs; x=np.exp(2j*np.pi*(off+1e6)*t); y=crop_band(x,fs,2400e6,2401)[1000:-1000]; amp=float(20*np.log10(np.sqrt(np.mean(abs(y)**2)))); freq=float(np.angle(np.vdot(y[:-1],y[1:]))*FS_OUT/(2*np.pi)); err=abs(freq-off)
            tests.append(dict(fs=fs,offset=off,amplitude_db=amp,freq_error_hz=err,passed=(abs(amp)<.1 and err<1000) if abs(off)==20e6 else amp<=-60))
    dump('crop_synthetic_tests.json',tests)
    fig,axs=plt.subplots(2,1,figsize=(12,7));axs[0].scatter(range(len(validations)),[v['residual_db'] for v in validations],s=3);axs[0].axhline(3,c='r');axs[0].set_ylabel('Residual over local floor (dB)');axs[1].scatter(range(len(validations)),[v['off_line_delta_db'] for v in validations],s=3);axs[1].set_ylabel('Off-line power change (dB)');axs[1].set_xlabel('Validation case');fig.tight_layout();fig.savefig(OUT/'cancel_examples.png',dpi=150);plt.close(fig)
    pairs=collections.defaultdict(set)
    for s in sources:
        if s['dataset']=='v3':pairs[re.sub(r'_s\d{4}_new','',s['path'])].add(s['group_id'])
    after=snapshot();dump('raw_snapshot_after.json',after)
    pngs=list(OUT.glob('*.png')); persistent=[l for l in inventory if l['presence_rate']>=.9]; covered=all(any(v['group']==l['group'] and abs(v['freq_mhz']-l['freq_mhz'])<=.02 for v in validations) for l in persistent)
    # Manifest duplicates are retained to meet explicit per-file exclusion requirement; report the counting conflict.
    nondup=sum(m['duplicate']=='False' for m in manifest)
    verify=dict(source_rows=dict(pass_=len(sources)==nondup+len(meta),actual=len(sources),expected=nondup+len(meta),duplicate_rows=len(manifest)-nondup,empty_groups=sum(not s['group_id'] for s in sources)),pairs=dict(pass_=all(len(g)==1 for g in pairs.values()),checked=len(pairs)),crop=dict(pass_=all(t['passed'] for t in tests),cases=tests),persistent_coverage=dict(pass_=covered,count=len(persistent)),scan=scancheck,raw_unchanged=dict(pass_=before==after,before=before,after=after),png=dict(pass_=all(p.stat().st_size>30000 for p in pngs),count=len(pngs)),minimum_duration=dict(pass_=all(r['measured_ms']>=200 for r in records.values()),minimum_ms=min(r['measured_ms'] for r in records.values())))
    dump('verification.json',verify);dump('recordings.json',[{k:v for k,v in r.items() if k!='files'} for r in records.values()])
    report=['# 截频带前谱线与同源核查','仅分析和规划；未生成训练样本、未切片保存、未训练、未划分。原始数据只读。',f'数据：{len(records)} 个录制/会话，{len(inventory)} 条组内合并细线；详细数值见 lines_inventory.csv，逐录制重叠见 lines_by_recording.csv。','谱估计：65536 Hann，50% 重叠；局部 ±1 MHz 中位数，突出 ≥10 dB、−10 dB 宽 ≤50 kHz；组内 20 kHz 合并。','抽样：均匀取 5 文件，确保 ≥200 ms（原要求 v3 两文件仅约100 ms，故优先时长）。出现率使用全部10 ms整块。','占用：v3 使用1 ms、4096 FFT、4 bin折叠、cell max；floor为各bin第10百分位在掩膜内的中位数；data沿用缓存。','重叠估计排除线±150kHz，以相邻150–300kHz占用推断线下宽带信号；该估计存在边界不确定性。']
    for group in specs:
        ls=[l for l in inventory if l['group']==group]; report.append(group+': '+('; '.join(f'{l["freq_mhz"]:.6f}MHz fc={l["is_at_fc"]} presence={l["presence_rate"]:.2f}' for l in ls) or '未检出满足判据的线'))
    report+=['消除参数：notch 为复数下变频后6阶Butter高通，零相位双向，半宽max(2×线宽,60kHz)；cancel为10ms块，低通降采样搜索、8倍零填充FFT抛物插值，再最小二乘正弦相减；零频另试复均值。',f'验证共{len(validations)}项；达标{sum(v["passed"] for v in validations)}；失败项全部保留cancel_validation.csv（残留≤3dB且离线功率变化≤0.1dB）。每个录制每线抽3个10ms块；不存在的重叠类别不伪造。','未达标的线禁止直接视为已消除；漂移/调制导致的残留需在F之前复核，当前未自动缩块调优。','固定输出61.44MS/s，通带±28MHz、过渡到±30.72MHz；90dB Kaiser FIR多相降采样。合成测试见crop_synthetic_tests.json。',f'截取中心范围：{min(p["center_mhz"] for p in plans if p["center_mhz"])}–{max(p["center_mhz"] for p in plans if p["center_mhz"])} MHz；逐项见crop_plan.csv。','占用比例<0.9：'+', '.join(r['key'] for r in records.values() if r['label']=='drone' and r['occupancy_fraction'] is not None and r['occupancy_fraction']<.9),'背景候选中心数和覆盖范围见background_sessions.csv与crop_plan.csv；no_valid_center：'+', '.join(b['session'] for b in backgrounds if not b['candidate_count']),f'同源表{len(sources)}行，{len(set(s["group_id"] for s in sources))}组；data所有epoch/seg同组，v3同raw session同组，扫频按中心频率细分。','之后训练/验证/测试以group_id为最小单位；同组所有片段（包括重叠片段）必须进入同一集合。','raw不单独入表；20260709-030546_ambient、20260709-040756_mavic2根目录副本与urban内重复；clean_export为空（任务给定）。',f'扫频对应验证：相关{correlation:.8f}，100MS/s坐标偏移{offset}点；'+('通过。' if scancheck['passed'] else '未通过，已排除扫频。'),f'行数冲突：任务要求保留duplicate排除行，但验收行数排除duplicate；本次保留全部{len(manifest)-nondup}个duplicate行并将该验收判fail，不隐去冲突。',f'验证详见verification.json；原始树前后文件数/最新mtime一致={before==after}；PNG数{len(pngs)}。']
    if len(report)>80:report=report[:7]+[' | '.join(report[i:i+3]) for i in range(7,7+len(specs),3)]+report[7+len(specs):]
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
if __name__=='__main__':main()
