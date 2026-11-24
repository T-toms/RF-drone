import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import csv,json,time,hashlib,sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path('C:/Users/Administrator/Desktop/数据集处理')
OUT=ROOT/'work/fusionB_20261110'
CACHE=Path('E:/fusionB_features_20261110')
NAMES=['shape_factor','log_kurt_a','papr999_db','c42n','c40n_abs','c41n_abs','c20n_abs','log_gamma_max','sigma_f','spec_flatness','occ_frac','duty','log1p_n_bursts','ctl_abs_power_db','ctl_floor_db','ctl_dyn_range_db','ctl_dc_rel','ctl_iq_power_ratio_db','ctl_iq_corr']
PROTECTED=['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109','train_stage1_L20_bgA_20261109','side_5p8_controls_20261109']
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def writecsv(p,rr):
    if not rr:return
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        fields=list(dict.fromkeys(k for r in rr for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def snapshot():
    result={}
    for d in PROTECTED:
        for p in sorted((ROOT/'work'/d).rglob('*')):
            if p.is_file():
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
                result[str(p.relative_to(ROOT))]=h.hexdigest()
    return result
def inputs():
    pp=sorted(Path('E:/drone_lab_spec_L20_20261108').glob('*_spec_meta.csv'))
    pp.append(Path('E:/drone_lab_spec_bgA_L20_20261109/bgA_meta.csv'))
    return {p.stem:rows(p) for p in pp}
NOTCH=rows(ROOT/'work/crop_slice_20261107/notch_table.csv')
def config(band):
    fs,fc,lo,hi=(61.44e6,2430,2405,2455) if band=='2.4GHz' else (100e6,5800,5755,5845)
    freq=fc+(np.arange(1024)-512)*fs/1024/1e6
    mask=(freq>=lo)&(freq<=hi)
    for r in NOTCH:
        if r['band']==band:mask &= abs(freq-float(r['freq_mhz']))>0.3
    return fs,int(round(fs*.0001)),mask
def feature(z,band):
    # Production: bounded PSD batches; all arithmetic complex128/float64.
    z=np.asarray(z,dtype=np.complex128);fs,L,mask=config(band);n=len(z)
    with np.errstate(all='ignore'):
        m=z.mean();raw=np.abs(z)**2;w=z-m;p0=np.mean(np.abs(w)**2);x=w/np.sqrt(p0)
        a=np.abs(x);p=a*a;ma=a.mean();v=(a-ma)**2;s2=np.mean(x*x)
        ps=np.zeros(1024);win=np.hanning(1024);frames=x[:n//1024*1024].reshape(-1,1024)
        for k in range(0,len(frames),128):ps+=np.sum(np.abs(np.fft.fft(frames[k:k+128]*win,axis=1))**2,axis=0)
        ps=np.fft.fftshift(ps/len(frames))[mask];db=10*np.log10(ps)
        fp=p[:n//L*L].reshape(-1,L).mean(1);on=fp>np.percentile(fp,10)*10**.6
        strong=(a[1:]>1)&(a[:-1]>1);inst=np.angle(x[1:]*x[:-1].conj())[strong]/(2*np.pi)
        fraw=raw[:n//L*L].reshape(-1,L).mean(1);LL=10*L
        db1=10*np.log10(raw[:n//LL*LL].reshape(-1,LL).mean(1));c42=np.mean(p*p)-abs(s2)**2-2
        re=w.real;im=w.imag
        result=[1/ma,np.log(np.mean(v*v)/np.mean(v)**2),10*np.log10(np.percentile(p,99.9)),np.sign(c42)*np.log1p(abs(c42)),np.log1p(abs(np.mean((x*x)**2)-3*s2*s2)),np.log1p(abs(np.mean(x*x*p)-3*s2)),abs(s2),np.log(np.max(np.abs(np.fft.rfft(a/ma-1)[1:])**2)/n),np.std(inst) if len(inst)>=1000 else np.nan,np.exp(np.mean(np.log(ps)))/np.mean(ps),np.mean(db>np.median(db)+10),on.mean(),np.log1p(np.sum(on[1:]&~on[:-1])),10*np.log10(raw.mean()),10*np.log10(np.percentile(fraw,10)),np.percentile(db1,90)-np.percentile(db1,10),abs(m)/np.sqrt(raw.mean()),10*np.log10(np.mean(re*re)/np.mean(im*im)),np.mean(re*im)/np.sqrt(np.mean(re*re)*np.mean(im*im))]
    result=np.array(result,dtype=np.float64);result[~np.isfinite(result)]=np.nan
    return result
def reference(z,band):
    # Deliberately direct, independent full-array reference; no production calls.
    z=np.array(z,dtype=np.complex128);N=z.size
    fs=61440000. if band=='2.4GHz' else 100000000.;fc=2430 if band=='2.4GHz' else 5800
    fsize=int(fs/10000);mean=np.mean(z);w=z-mean;x=w/np.sqrt(np.mean(w.real**2+w.imag**2));a=np.absolute(x)
    ampmean=np.mean(a);center=a-ampmean;moment2=np.mean(x**2);moment42=np.mean(a**4)-abs(moment2)**2-2
    psd=np.mean(abs(np.fft.fft(x[:N//1024*1024].reshape(-1,1024)*np.hanning(1024),axis=1))**2,axis=0)
    psd=np.fft.fftshift(psd);freq=fc+np.fft.fftshift(np.fft.fftfreq(1024,1/fs))/1e6
    valid=(freq>= (2405 if band=='2.4GHz' else 5755))&(freq<= (2455 if band=='2.4GHz' else 5845))
    for r in NOTCH:
        if r['band']==band:valid &= abs(freq-float(r['freq_mhz']))>0.3
    psd=psd[valid];power=np.mean(abs(x[:N//fsize*fsize].reshape(-1,fsize))**2,axis=1)
    active=power>np.percentile(power,10)*10**.6;gate=(a[:-1]>1)&(a[1:]>1)
    freqinst=np.angle(x[1:]*np.conjugate(x[:-1]))*fs/(2*np.pi)
    rawpower=np.mean(abs(z[:N//fsize*fsize].reshape(-1,fsize))**2,axis=1)
    size1=10*fsize;db1=10*np.log10(np.mean(abs(z[:N//size1*size1].reshape(-1,size1))**2,axis=1))
    with np.errstate(all='ignore'):
        out=[1/ampmean,np.log(np.mean(center**4)/np.mean(center**2)**2),10*np.log10(np.percentile(a**2,99.9)),np.sign(moment42)*np.log(1+abs(moment42)),np.log(1+abs(np.mean(x**4)-3*moment2**2)),np.log(1+abs(np.mean(x**3*np.conjugate(x))-3*moment2)),abs(moment2),np.log(np.max(abs(np.fft.fft(a/ampmean-1)[1:])**2)/N),np.std(freqinst[gate])/fs if gate.sum()>=1000 else np.nan,np.exp(np.mean(np.log(psd)))/np.mean(psd),np.mean(10*np.log10(psd)>np.median(10*np.log10(psd))+10),np.mean(active),np.log1p(np.count_nonzero(np.diff(active.astype(int))==1)),10*np.log10(np.mean(abs(z)**2)),10*np.log10(np.percentile(rawpower,10)),np.percentile(db1,90)-np.percentile(db1,10),abs(mean)/np.sqrt(np.mean(abs(z)**2)),10*np.log10(np.mean(w.real**2)/np.mean(w.imag**2)),np.corrcoef(w.real,w.imag)[0,1]]
    out=np.array(out);out[~np.isfinite(out)]=np.nan;return out
def checks(data):
    selected=[]
    for band in ['2.4GHz','5.8GHz']:
        rr=[r for name,rs in data.items() if 'rejected' not in name for r in rs if r['band']==band]
        for dataset,label in [('data','drone'),('v3','drone'),('v3','background'),('dataA_mirror','background')]:
            selected += [r for r in rr if r['dataset']==dataset and r['label']==label][:3]
    comparisons=[]
    for r in sorted(selected,key=lambda r:(r['block_file'],int(r['start']))):
        z=np.load(r['block_file'],mmap_mode='r')[int(r['start']):int(r['start'])+int(r['n_samples'])]
        a=feature(z,r['band']);b=reference(z,r['band']);delta=abs(a-b);rel=delta/np.maximum(abs(b),1e-300)
        assert np.all((delta<=1e-6)|(rel<=1e-4)|(np.isnan(a)&np.isnan(b))), (r['sample_id'],a,b)
        comparisons.append(dict(sample_id=r['sample_id'],band=r['band'],dataset=r['dataset'],label=r['label'],max_abs=float(np.nanmax(delta)),max_rel=float(np.nanmax(rel))))
    synth=[];rng=np.random.default_rng(2026)
    for band in ['2.4GHz','5.8GHz']:
        fs,L,_=config(band);N=200*L;noise=(rng.standard_normal(N)+1j*rng.standard_normal(N))/np.sqrt(2)
        tone=np.exp(2j*np.pi*np.arange(N)*.125);burst=noise.copy();burst[40*L:60*L]*=np.sqrt(101)
        for name,z in [('white',noise),('tone',tone),('burst',burst)]:
            f=feature(z,band);synth.append(dict(band=band,signal=name,**dict(zip(NAMES,f.tolist()))))
            if name=='white':assert abs(f[3])<.01 and f[6]<.01 and f[9]>.95 and f[11]<.05
            if name=='tone':assert abs(f[0]-1)<1e-8 and abs(f[3]+np.log(2))<1e-8
            if name=='burst':assert abs(f[11]-.1)<.02
    writecsv(OUT/'check_dual.csv',comparisons);writecsv(OUT/'check_synthetic.csv',synth)
    dump(OUT/'checks.json',dict(passed=True,n_real=len(comparisons),max_abs=max(r['max_abs'] for r in comparisons),max_rel=max(r['max_rel'] for r in comparisons)))
def extract(data):
    CACHE.mkdir(exist_ok=True);dump(CACHE/'feature_names.json',NAMES)
    arrays={};tasks=defaultdict(list)
    for name,rr in data.items():
        arrays[name]=np.lib.format.open_memmap(CACHE/(name+'_features.npy'),mode='w+',dtype='float64',shape=(len(rr),19))
        writecsv(CACHE/(name+'_features_meta.csv'),[dict(r,row_index=i) for i,r in enumerate(rr)])
        for i,r in enumerate(rr):tasks[r['block_file']].append((int(r['start']),name,i,r))
    done=0;readbytes=0;t=time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path,jobs in sorted(tasks.items()):
            block=np.load(path,mmap_mode='r');jobs.sort();pending=[]
            for start,name,i,r in jobs:
                z=np.array(block[start:start+int(r['n_samples'])]);readbytes+=z.nbytes
                pending.append((name,i,pool.submit(feature,z,r['band'])))
                if len(pending)>=8:
                    for nm,j,f in pending:arrays[nm][j]=f.result();done+=1
                    pending=[]
            for nm,j,f in pending:arrays[nm][j]=f.result();done+=1
            if done%100< len(jobs):print(f'extract {done}/{sum(map(len,data.values()))} elapsed={time.time()-t:.1f}s',flush=True)
            for a in arrays.values():a.flush()
    rng=np.random.default_rng(2026);align=[]
    for name,rr in data.items():
        saved=rows(CACHE/(name+'_features_meta.csv'))
        for i in rng.choice(len(rr),min(20,len(rr)),replace=False):
            assert saved[i]['sample_id']==rr[i]['sample_id'] and int(saved[i]['row_index'])==i
            align.append(dict(name=name,row_index=int(i),sample_id=rr[i]['sample_id']))
    writecsv(OUT/'check_alignment.csv',align);dump(OUT/'extraction.json',dict(seconds=time.time()-t,read_bytes=readbytes,rows={n:len(r) for n,r in data.items()},aligned=len(align)))
def noise_selection(data):
    rng=np.random.default_rng(2026);chosen=[]
    for band in ['2.4GHz','5.8GHz']:
        rr=[dict(r,meta_name=name,row_index=i) for name,rs in data.items() if 'rejected' not in name for i,r in enumerate(rs) if r['band']==band and r['split']=='test' and r.get('keep','True')=='True']
        dd=[r for r in rr if r['dataset']=='data' and r['label']=='drone'];groups=defaultdict(list)
        for r in dd:groups[r['model']].append(r)
        # Proportional stratified sample, exact budget with largest remainder.
        budget=min(400,len(dd));quota={m:budget*len(g)/len(dd) for m,g in groups.items()};counts={m:int(q) for m,q in quota.items()}
        for m in sorted(groups,key=lambda m:quota[m]-counts[m],reverse=True)[:budget-sum(counts.values())]:counts[m]+=1
        for m,g in sorted(groups.items()):chosen.extend(g[i] for i in rng.choice(len(g),counts[m],replace=False))
        chosen.extend(r for r in rr if r['dataset']!='data')
    chosen.sort(key=lambda r:(r['block_file'],int(r['start'])))
    return chosen
def noise(data):
    chosen=noise_selection(data);out=np.empty((len(chosen),5,19));t=time.time();readbytes=0
    def compute(z,r,seed):
        result=[np.load(CACHE/(r['meta_name']+'_features.npy'),mmap_mode='r')[int(r['row_index'])].copy()]
        power=np.mean(abs(z.astype(np.complex128))**2)
        for level,snr in enumerate([0,-5,-10,-15]):
            rand=np.random.default_rng(np.random.SeedSequence([seed,level]));no=(rand.standard_normal(len(z))+1j*rand.standard_normal(len(z)))*np.sqrt(power/10**(snr/10)/2)
            result.append(feature(z+no,r['band']))
        return result
    with ThreadPoolExecutor(max_workers=8) as pool:
        current=None;pending=[]
        for i,r in enumerate(chosen):
            if current!=r['block_file']:
                for j,f in pending:out[j]=f.result()
                pending=[];block=np.load(r['block_file'],mmap_mode='r');current=r['block_file']
            seed=int.from_bytes(hashlib.sha256(('2026|'+r['sample_id']).encode()).digest()[:8],'little');r['noise_seed']=str(seed)
            start=int(r['start']);z=np.array(block[start:start+int(r['n_samples'])]);readbytes+=z.nbytes
            pending.append((i,pool.submit(compute,z,r,seed)))
            if len(pending)>=12:
                for j,f in pending:out[j]=f.result()
                pending=[]
            if i%200==0:print(f'noise {i}/{len(chosen)} elapsed={time.time()-t:.1f}s',flush=True)
        for j,f in pending:out[j]=f.result()
    np.save(CACHE/'noise_features.npy',out);writecsv(CACHE/'noise_meta.csv',chosen);dump(OUT/'noise_runtime.json',dict(seconds=time.time()-t,read_bytes=readbytes,n=len(chosen),snr=['clean',0,-5,-10,-15]))
if __name__=='__main__':
    mode=sys.argv[1];OUT.mkdir(exist_ok=True)
    if mode=='snapshot':dump(OUT/'protected_before.json',snapshot())
    elif mode=='verify':
        after=snapshot();dump(OUT/'protected_after.json',after);assert after==json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'));print('PROTECTED IDENTICAL',len(after))
    elif mode=='checks':checks(inputs())
    elif mode=='extract':extract(inputs())
    elif mode=='noise':noise(inputs())
