"""Q2: calibration must finish and REPORT lock must exist before detection.
All matrices time x frequency; all intervals half open. No disk cache.
Run D:/anaconda/python.exe -I run_q2.py {calibrate,detect,finish,verify}.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['PYTHONIOENCODING']='utf-8'
import sys, json, time, csv, hashlib, importlib.util, collections, concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from scipy import fft, ndimage as ndi
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'work/side_Q_band_detect_pilot_20261111'
spec=importlib.util.spec_from_file_location('copied_io',OUT/'copied_io.py')
io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)
KS=(2,3,4,6); GS=(.5,2)
PROC=psutil.Process(); peak=0
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def load(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def old(name): return json.loads((OLD/name).read_text(encoding='utf-8'))
def resources():
    global peak
    peak=max(peak,PROC.memory_info().peak_wset)
    assert peak<8*1024**3,('memory limit',peak)
def tag(k,g): return f'k{k}_g{g:g}'
def grid(c,bounds=None,n=None,synthetic=None):
    n=n or (1024 if c['source']=='data' else 2048)
    fs=c['fs'];fc=c['fc'];x=io.read_iq(c) if synthetic is None else synthetic
    nt=len(x)//n
    a=x[:nt*n].reshape(nt,n)*np.hanning(n).astype(np.float32)
    z=fft.fftshift(fft.fft(a,axis=1,workers=1),axes=1)
    power=z.real*z.real+z.imag*z.imag
    f=fc+fft.fftshift(fft.fftfreq(n,1/(fs/1e6)))
    if bounds is None:
        bounds=(2395,2488.5) if fc<3000 else (5720,5855)
        lo=max(bounds[0],fc-fs/2e6+1.5);hi=min(bounds[1],fc+fs/2e6-1.5)
    else: lo,hi=bounds
    ix=(f>=lo)&(f<=hi);f=f[ix];power=power[:,ix]
    assert len(f)>17
    tones=[fc,2426.88,2457.6,5740,5795.5] if c['source']=='data' else [fc]
    bad=np.zeros(len(f),bool)
    for t in tones: bad|=abs(f-t)<=.3
    raw=10*np.log10(np.maximum(power,1e-30))
    # Fixed-tone power replaced before smoothing so it cannot leak to neighbours.
    preliminary=float(np.percentile(power[:,~bad],20));power[:,bad]=preliminary
    smooth=ndi.uniform_filter(power,size=(16,4),mode='nearest')
    S=10*np.log10(np.maximum(smooth,1e-30))
    nf=np.percentile(S,20,axis=0);global_floor=float(np.median(nf[~bad]))
    nf[bad]=global_floor
    floor=ndi.median_filter(np.minimum(nf,global_floor+3),size=17,mode='nearest')
    S[:,bad]=floor[bad]; raw[:,bad]=floor[bad]
    # Q occupancy reference: time-mean power q10 floor, 1ms maximum +10dB.
    per=int(round(fs/1000))//n;nc=nt//per
    bp=power[:nc*per].reshape(nc,per,-1)
    mean_db=10*np.log10(np.maximum(bp.mean(1),1e-30))
    ofloor=float(np.median(np.percentile(mean_db[:,~bad],10,axis=0)))
    occ=(10*np.log10(np.maximum(bp.max(1),1e-30))>ofloor+10).mean(0);occ[bad]=0
    return dict(raw=raw,S=S,F=floor,f=f,bad=bad,occ=occ,dt=n/fs,lo=lo,hi=hi)
def detect(d,k):
    mask=d['S']>d['F']+k
    mask=ndi.binary_closing(mask,structure=np.ones((9,3),bool))
    mask=ndi.binary_opening(mask,structure=np.ones((3,2),bool))
    mask[:,d['bad']]=False
    lab,num=ndi.label(mask,structure=ndi.generate_binary_structure(2,1))
    areas=np.bincount(lab.ravel());ids=np.flatnonzero(areas>=50);ids=ids[ids!=0]
    if not len(ids): return []
    excess=d['S']-d['F'];sums=ndi.sum(excess,lab,index=ids);maxima=ndi.maximum(excess,lab,index=ids)
    objects=ndi.find_objects(lab);f=d['f'];df=f[1]-f[0];rows=[]
    for idx,s,mx in zip(ids,sums,maxima):
        t,fr=objects[idx-1]
        rows.append(dict(box_id=len(rows)+1,t_start_ms=t.start*d['dt']*1000,t_end_ms=t.stop*d['dt']*1000,
          f_start_mhz=float(f[fr.start]-df/2),f_end_mhz=float(f[fr.stop-1]+df/2),area_px=int(areas[idx]),
          mean_excess_db=float(s/areas[idx]),max_excess_db=float(mx),t0=t.start,t1=t.stop,f0=fr.start,f1=fr.stop))
    return rows
def cal_clips():
    # Only 2400MHz recordings overlap the specified 2343--2393 noise range.
    a=[dict(c,cal_group='a',cal_bounds=(2343,2393),cal_n=1024) for c in old('clips.json') if c['source']=='data' and c['fc']==2400]
    root=Path('G:/drone/raw/2026-07-09_urban/ambient/2026-07-16_lab/20260715-075302_ambient')
    b=[]
    for p in sorted(root.glob('*.json')):
        m=json.loads(p.read_text(encoding='utf-8'));cap=m['capture']
        if abs(cap['center_freq_hz']/1e6-2367.06)>.01: continue
        fs=cap['sample_rate_hz'];length=round(fs*.02)
        for slot,start in enumerate((0,(cap['sample_count']-length)//2)):
            b.append(dict(clip_id=f'B{len(b):03d}',source='raw',fc=cap['center_freq_hz']/1e6,fs=fs,length=length,start=start,
                parts=[dict(path=str(p.with_suffix('.bin')),start=start,count=length)],cal_group='b',cal_bounds=(2345,2395),cal_n=1024))
    assert len(a)==342 and len(b)==10
    c=[]
    for source,fs,fc,n,bounds in [('data',122880000,2400,1024,(2343,2393)),('raw',250000000,2440,2048,(2395,2488.5))]:
        for slot in range(60): c.append(dict(clip_id=f'C{len(c):03d}',source=source,fc=fc,fs=fs,length=round(fs*.02),cal_group='c',cal_bounds=bounds,cal_n=n))
    return a+b+c
def calibration():
    assert not (OUT/'threshold_lock.json').exists(),'calibration already locked'
    dump('protected_before.json',io.snapshot())
    clips=cal_clips();dump('calibration_clips.json',clips)
    rows=[];rng=np.random.default_rng(20261111);start=time.time()
    for i,c in enumerate(clips):
        x=None
        if c['cal_group']=='c': x=(rng.standard_normal(c['length'],dtype=np.float32)+1j*rng.standard_normal(c['length'],dtype=np.float32))/np.sqrt(2)
        d=grid(c,c['cal_bounds'],c['cal_n'],x)
        for k in KS:
            boxes=detect(d,k);bands=io.merge(boxes,.5,len(d['S']))
            rows.append(dict(cal_group=c['cal_group'],clip_id=c['clip_id'],grid=f"{c['fs']}/{c['cal_n']}",k=k,g=.5,
              band_count=len(bands),box_count=len(boxes),width_mhz=sum(b['width_mhz'] for b in bands),analysis_width_mhz=d['hi']-d['lo']))
        resources()
        if i%20==0: print('calibration',i+1,len(clips),round(time.time()-start,1),flush=True)
    df=pd.DataFrame(rows);df['width_fraction']=df.width_mhz/df.analysis_width_mhz;df.to_csv(OUT/'calibration.csv',index=False,encoding='utf-8-sig')
    dump('calibration_compute.json',dict(seconds=time.time()-start,peak_bytes=peak))
    if '--hold-lock' in sys.argv:
        print('CALIBRATION COMPUTED; threshold lock awaits coordinator clarification',flush=True)
        return
    lock_calibration()
def lock_calibration():
    df=pd.read_csv(OUT/'calibration.csv');computed=load('calibration_compute.json')
    stats=[]
    for (group,k),v in df.groupby(['cal_group','k']):
        s=dict(cal_group=group,k=int(k),n=len(v),median=v.band_count.median(),p95=v.band_count.quantile(.95),maximum=int(v.band_count.max()),
            total_width_fraction=v.width_mhz.sum()/v.analysis_width_mhz.sum(),width_fraction_max=v.width_fraction.max())
        s['pass']=bool(s['median']==0 and s['p95']<=1 and s['total_width_fraction']<=.005);stats.append(s)
    pd.DataFrame(stats).to_csv(OUT/'stats_calibration.csv',index=False,encoding='utf-8-sig')
    eligible=[k for k in KS if all(s['pass'] for s in stats if s['k']==k)];ks=min(eligible) if eligible else 6
    lines=['# Q2 自动找信号频带试验（2026-11-11）','',f'纯噪声校准先于无人机检测；k* = {ks} dB；满足全部条件的候选 k = {eligible}。',
      'a组仅含中心2400MHz的57次录制×6段=342段；中心5800MHz的234段不覆盖指定2343–2393MHz，未纳入校准。',
      '协调者确认：5.8GHz实验室接收机没有空闲频段（任务I），未用5.8GHz真实接收机噪声单独校准。',
      'k*同样用于5.8GHz；按逐频率本地底噪相对定义，假设平滑噪声起伏由平均点数决定且与接收机无关；合成噪声覆盖两种网格。这是假设，不是验证。',
      'b组5个125MS/s文件各取[0,20)与[50,70)ms，共10段；c组两网格各60段，种子20261111。',
      '宽度条件按各组误检总宽度/分析总宽度计算；另在CSV记录每段比例和组内最大值。',
      '|组|k dB|段数|误检带数中位/P95/最大|误检总宽度占比|单段宽度占比最大|通过|','|---|---:|---:|---:|---:|---:|---|']
    for s in stats: lines.append(f"|{s['cal_group']}|{s['k']}|{s['n']}|{s['median']:g}/{s['p95']:g}/{s['maximum']}|{s['total_width_fraction']:.6%}|{s['width_fraction_max']:.6%}|{s['pass']}|")
    if not eligible: lines.append('四个候选均未满足全部条件，按规范取k*=6 dB。')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump('threshold_lock.json',dict(k_star=ks,eligible=eligible,locked_at=time.strftime('%Y-%m-%d %H:%M:%S'),calibration_seconds=computed['seconds'],
       report_sha256=hashlib.sha256((OUT/'REPORT.md').read_bytes()).hexdigest(),peak_bytes=computed['peak_bytes']))
    print('THRESHOLD LOCKED',ks,flush=True)
BOXKEYS=['clip_id','box_id','t_start_ms','t_end_ms','f_start_mhz','f_end_mhz','area_px','mean_excess_db','max_excess_db']
BANDKEYS=['clip_id','band_id','f_start_mhz','f_end_mhz','width_mhz','time_duty','box_count','max_excess_db']
META=['clip_id','source','band','model','state','bw','label','group','date','fc','start']
SEGKEYS=META+['coverage','reference_total','band_count','box_count','band_fraction','width_mhz','analysis_width_mhz','frames','bins']
def detection():
    lock=load('threshold_lock.json');ks=lock['k_star'];assert (OUT/'REPORT.md').exists()
    assert hashlib.sha256((OUT/'REPORT.md').read_bytes()).hexdigest()==lock['report_sha256'],'initial calibration report changed'
    assert not (OUT/'detection_complete.json').exists(),'already complete'
    clips=old('clips.json');refs={float(fc):np.array(v['occupancy_p90']) for fc,v in old('background_reference.json').items()}
    handles=[];writers={};start=time.time()
    for k in KS:
        for prefix,keys in [('boxes',BOXKEYS)] if k==ks else []:
            fh=(OUT/f'boxes_k{k}.csv').open('w',encoding='utf-8-sig',newline='');handles.append(fh);w=csv.DictWriter(fh,fieldnames=keys,extrasaction='ignore');w.writeheader();writers[(prefix,k,0)]=w
        for g in GS:
            for prefix,keys in [('bands',BANDKEYS),('segments',SEGKEYS)]:
                fh=(OUT/f'{prefix}_{tag(k,g)}.csv').open('w',encoding='utf-8-sig',newline='');handles.append(fh)
                w=csv.DictWriter(fh,fieldnames=keys,extrasaction='ignore');w.writeheader();writers[(prefix,k,g)]=w
    executor=concurrent.futures.ThreadPoolExecutor(max_workers=4)
    for i,c in enumerate(clips):
        d=grid(c)
        if c['source']=='data':
            weights=np.where(d['S']>d['F']+10,np.maximum(10.**((d['S']-d['F'])/10)-1,0)*10.**(d['F']/10),0).sum(0,dtype=np.float64)
        elif c['label']=='drone': weights=np.maximum(d['occ']-refs[c['fc']],0)
        else: weights=np.zeros(len(d['f']))
        total=float(weights.sum());meta={key:c[key] for key in META}
        results=list(executor.map(lambda k:detect(d,k),KS))
        for k,boxes in zip(KS,results):
            if k==ks: writers[('boxes',k,0)].writerows(dict(b,clip_id=c['clip_id']) for b in boxes)
            for g in GS:
                bands=io.merge(boxes,g,len(d['S']));covered=np.zeros(len(d['f']),bool)
                for b in bands: covered|=(d['f']>=b['f_start_mhz'])&(d['f']<b['f_end_mhz'])
                width=sum(b['width_mhz'] for b in bands)
                writers[('bands',k,g)].writerows(dict(b,clip_id=c['clip_id']) for b in bands)
                writers[('segments',k,g)].writerow(dict(meta,coverage=float(weights[covered].sum()/total) if total>0 else float('nan'),reference_total=total,
                   band_count=len(bands),box_count=len(boxes),band_fraction=width/(d['hi']-d['lo']),width_mhz=width,analysis_width_mhz=d['hi']-d['lo'],frames=len(d['S']),bins=len(d['f'])))
        for fh in handles: fh.flush()
        resources()
        if i%20==0: print('detection',i+1,len(clips),'seconds',round(time.time()-start,1),'peakGiB',round(peak/1024**3,3),flush=True)
    for fh in handles: fh.close()
    executor.shutdown()
    dump('detection_complete.json',dict(clips=len(clips),seconds=time.time()-start,peak_bytes=peak))
if __name__=='__main__':
    if sys.argv[1]=='calibrate': calibration()
    elif sys.argv[1]=='lock': lock_calibration()
    elif sys.argv[1]=='detect': detection()
