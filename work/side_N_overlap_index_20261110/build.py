"""Task N: read-only source/cache access, half-overlap indexes, complete audit.
Run with the workspace .venv Python -I; no source-module import or execution.
"""
import csv, json, hashlib, collections, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
O = Path(__file__).resolve().parent
F = ROOT/'work/crop_slice_20261107'
LS = [10,20,30,40,50]
FLAGS = ['lack_feature','primary_weak','power_shift','floor_raised','interference']
FS = {'2.4GHz':61440000,'5.8GHz':100000000}
FC = {'2.4GHz':2430,'5.8GHz':5800}
WIDTH = {'2.4GHz':25,'5.8GHz':45}
PROTECTED = ['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109','side_L_wifimix_20261110','side_M_notchfill_20261110']

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def dump(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def write(p,rs,fields=None):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rs[0])); w.writeheader(); w.writerows(rs)

def snapshot():
    result={}
    for name in PROTECTED:
        base=ROOT/'work'/name
        if not base.is_dir(): raise RuntimeError('Missing protected directory: '+str(base))
        for p in sorted(base.rglob('*')):
            if p.is_file():
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
                result[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    return result

def window_data(ss,act,ao,mx,ix,floor):
    av=np.array([act[s:e].mean(0) for s,e,_,_ in ss]); ov=np.array([ao[s:e].mean() for s,e,_,_ in ss]); sv=np.full((len(ss),len(ix)),np.nan)
    for i,(s,e,_,_) in enumerate(ss):
        for k,b in enumerate(ix):
            block=mx[s:e][:,b]; active=(block>floor+10).any(1)
            if active.sum()>=2: sv[i,k]=np.median(block[active].mean(1)-floor)
    return av,ov,sv

NOTCHES = rows(F/'notch_table.csv')
for r in NOTCHES:
    for k in ['freq_mhz','half_width_khz']: r[k]=float(r[k])
c=SimpleNamespace(FS=FS,FC=FC,WIDTH=WIDTH,NOTCHES=NOTCHES)

def intervals(mask):
 e=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))
def screening_original(gb,ar):
 band=gb[0]['band'];fo=c.FS[band];nfft=ar[0]['mean'].shape[1];freq=np.fft.fftshift(np.fft.fftfreq(nfft,1/fo))/1e6+c.FC[band];mask=abs(freq-c.FC[band])<=c.WIDTH[band];env=np.zeros(nfft,bool)
 for r in c.NOTCHES:
  if r['band']==band:env |= abs(freq-r['freq_mhz'])<=r['half_width_khz']/1000
 mask &= ~env;means=np.concatenate([a['mean'] for a in ar]);mx=np.concatenate([a['max'] for a in ar]);floor=float(np.median(np.percentile(means[:,mask],10,axis=0)));occ=(mx>floor+10).mean(0);occupied=mask&(occ>=.05)
 # Same v3b adjacent-gap merge, with new frequency grid, after clipping fixed band.
 runs=intervals(occupied);merged=[]
 for lo,hi in runs:
  if merged and np.count_nonzero(mask[merged[-1][1]:lo])<=2:merged[-1]=(merged[-1][0],hi)
  else:merged.append((lo,hi))
 ix=[]
 for lo,hi in merged:
  b=np.zeros(nfft,bool);b[lo:hi]=True;b &= mask
  if b.any():ix.append(b)
 floor_cells=np.percentile(means[:,mask],10,axis=1);act=np.stack([(mx[:,b]>floor+10).mean(1) for b in ix],axis=1) if ix else np.empty((len(mx),0));outside=mask & ~(np.logical_or.reduce(ix) if ix else False);ao=(mx[:,outside]>floor+10).mean(1) if outside.any() else np.zeros(len(mx));spans={};offset=0
 for b,a in zip(gb,ar):
  for L in LS:
   spans.setdefault(L,[]).extend((offset+j,offset+j+L,b,j) for j in range(0,len(a['mean'])-L+1,L))
  offset+=len(a['mean'])
 data={}
 for L,ss in spans.items():
  av=np.array([act[s:e].mean(0) for s,e,_,_ in ss]);ov=np.array([ao[s:e].mean() for s,e,_,_ in ss]);sv=np.full((len(ss),len(ix)),np.nan)
  for i,(s,e,_,_) in enumerate(ss):
   for k,b in enumerate(ix):
    block=mx[s:e][:,b];active=(block>floor+10).any(1)
    if active.sum()>=2:sv[i,k]=np.median(block[active].mean(1)-floor)
  data[L]=(av,ov,sv)
 av50=data[50][0];occ50=(av50>=.01).mean(0) if len(av50) else np.zeros(len(ix));chars=np.flatnonzero(occ50>=.9);primary=int(max(chars,key=lambda k:np.median(av50[:,k])*ix[k].sum())) if len(chars) else (int(np.argmax(occ50)) if len(ix) else -1);result={}
 for L,ss in spans.items():
  av,ov,sv=data[L];rr=[]
  if not len(ss):result[L]=rr;continue
  dense=[k for k in chars if (av[:,k]>=.01).mean()>=.9];med=np.median(av,axis=0);meds=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in sv.T]);outlim=max(.05,3*np.median(ov));tested=set(([primary] if primary>=0 else [])+[k for k in dense if med[k]>=.2])
  for i,(s,e,b,j) in enumerate(ss):
   flags=[any(av[i,k]<.01 for k in dense),primary>=0 and av[i,primary]<.25*med[primary],any(abs(sv[i,k]-meds[k])>6 for k in tested),np.median(floor_cells[s:e])-floor>3,ov[i]>outlim];rr.append((b,j,[bool(v) for v in flags]))
  result[L]=rr
 return result,dict(group_id=gb[0]['group_id'],floor_db=floor,n_subbands=len(ix),subbands=[dict(lo=float(freq[b][0]),hi=float(freq[b][-1])) for b in ix],primary=primary,persistent=chars.tolist())
def screening(gb,ar):
 band=gb[0]['band'];fo=c.FS[band];nfft=ar[0]['mean'].shape[1];freq=np.fft.fftshift(np.fft.fftfreq(nfft,1/fo))/1e6+c.FC[band];mask=abs(freq-c.FC[band])<=c.WIDTH[band];env=np.zeros(nfft,bool)
 for r in c.NOTCHES:
  if r['band']==band:env |= abs(freq-r['freq_mhz'])<=r['half_width_khz']/1000
 mask &= ~env;means=np.concatenate([a['mean'] for a in ar]);mx=np.concatenate([a['max'] for a in ar]);floor=float(np.median(np.percentile(means[:,mask],10,axis=0)));occ=(mx>floor+10).mean(0);occupied=mask&(occ>=.05)
 # Same v3b adjacent-gap merge, with new frequency grid, after clipping fixed band.
 runs=intervals(occupied);merged=[]
 for lo,hi in runs:
  if merged and np.count_nonzero(mask[merged[-1][1]:lo])<=2:merged[-1]=(merged[-1][0],hi)
  else:merged.append((lo,hi))
 ix=[]
 for lo,hi in merged:
  b=np.zeros(nfft,bool);b[lo:hi]=True;b &= mask
  if b.any():ix.append(b)
 floor_cells=np.percentile(means[:,mask],10,axis=1);act=np.stack([(mx[:,b]>floor+10).mean(1) for b in ix],axis=1) if ix else np.empty((len(mx),0));outside=mask & ~(np.logical_or.reduce(ix) if ix else False);ao=(mx[:,outside]>floor+10).mean(1) if outside.any() else np.zeros(len(mx));spans={};offset=0
 for b,a in zip(gb,ar):
  for L in LS:
   spans.setdefault(L,[]).extend((offset+j,offset+j+L,b,j) for j in range(0,len(a['mean'])-L+1,L))
  offset+=len(a['mean'])
 data={}
 for L,ss in spans.items():
  av=np.array([act[s:e].mean(0) for s,e,_,_ in ss]);ov=np.array([ao[s:e].mean() for s,e,_,_ in ss]);sv=np.full((len(ss),len(ix)),np.nan)
  for i,(s,e,_,_) in enumerate(ss):
   for k,b in enumerate(ix):
    block=mx[s:e][:,b];active=(block>floor+10).any(1)
    if active.sum()>=2:sv[i,k]=np.median(block[active].mean(1)-floor)
  data[L]=(av,ov,sv)
 av50=data[50][0];occ50=(av50>=.01).mean(0) if len(av50) else np.zeros(len(ix));chars=np.flatnonzero(occ50>=.9);primary=int(max(chars,key=lambda k:np.median(av50[:,k])*ix[k].sum())) if len(chars) else (int(np.argmax(occ50)) if len(ix) else -1);result={}
 for L,ss in spans.items():
  av,ov,sv=data[L];rr=[]
  if not len(ss):result[L]=rr;continue
  dense=[k for k in chars if (av[:,k]>=.01).mean()>=.9];med=np.median(av,axis=0);meds=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in sv.T]);outlim=max(.05,3*np.median(ov));tested=set(([primary] if primary>=0 else [])+[k for k in dense if med[k]>=.2])
  # All reference statistics above are computed exclusively on original spans.
  if L not in [30,40,50]: continue
  ss=[];offset=0
  for b,a in zip(gb,ar):
   ss.extend((offset+j,offset+j+L,b,j) for j in range(0,len(a['mean'])-L+1,L//2))
   offset+=len(a['mean'])
  av,ov,sv=window_data(ss,act,ao,mx,ix,floor)
  for i,(s,e,b,j) in enumerate(ss):
   flags=[any(av[i,k]<.01 for k in dense),primary>=0 and av[i,primary]<.25*med[primary],any(abs(sv[i,k]-meds[k])>6 for k in tested),np.median(floor_cells[s:e])-floor>3,ov[i]>outlim];rr.append((b,j,[bool(v) for v in flags]))
  result[L]=rr
 return result,dict(group_id=gb[0]['group_id'],floor_db=floor,n_subbands=len(ix),subbands=[dict(lo=float(freq[b][0]),hi=float(freq[b][-1])) for b in ix],primary=primary,persistent=chars.tolist())

def typed(rs):
    for r in rs:
        for k in ['start','n_samples']:r[k]=int(r[k])
        for k in FLAGS+['keep']:r[k]=r[k]=='True'
    return rs

def coverage(rs):
    by=collections.defaultdict(list)
    for r in rs:
        if r['keep']: by[(r['block_file'],r['band'])].append((r['start'],r['start']+r['n_samples']))
    total=0.
    for (_,band),ss in by.items():
        end=-1; count=0
        for s,e in sorted(ss):
            count+=max(0,e-max(s,end));end=max(end,e)
        total+=count/FS[band]
    return total

def require(ok,message):
    if not ok: raise AssertionError(message)

def main():
    before=snapshot(); dump(O/'sha256_before.json',before)
    print('HASH_BEFORE',len(before),flush=True)
    bs=json.loads((F/'plan.json').read_text(encoding='utf-8-sig'))
    blockrows={r['block_file']:r for r in rows(F/'blocks.csv')}
    gs={r['group_id']:r for r in rows(F/'group_split.csv')}
    original={L:typed(rows(F/f'index_L{L}.csv')) for L in LS}
    origmap={L:{(r['block_file'],r['start'],r['n_samples']):r for r in rr} for L,rr in original.items()}
    lengths={}; cells_count={}; by=collections.defaultdict(list)
    for b in bs:by[b['group_id']].append(b)
    output={L:[] for L in [30,40,50]}; refinfo=[]
    for gi,(gid,gb) in enumerate(by.items()):
        ars=[]
        for b in gb:
            cell=F/'cells'/Path(b['block_file']).stem
            ar={k:np.load(cell/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['mean','max']}
            require(all(not a.flags.writeable for a in ar.values()),'Cache not read only')
            z=np.load(b['block_file'],mmap_mode='r',allow_pickle=False)
            n=len(z);del z
            require(n==int(blockrows[b['block_file']]['n_samples']),'Block length differs from blocks.csv')
            require(len(ar['mean'])==len(ar['max'])==n//round(FS[b['band']]/1000),'Cell count differs from block length')
            lengths[b['block_file']]=n; cells_count[b['block_file']]=len(ar['mean']);ars.append(ar)
        if gb[0]['label']=='drone':
            rr,info=screening(gb,ars);refinfo.append(info)
        else:
            rr={L:[(b,j,[False]*5) for b,a in zip(gb,ars) for j in range(0,len(a['mean'])-L+1,L//2)] for L in output}
        for L in output:
            for b,j,flags in rr[L]:
                cs=round(FS[b['band']]/1000); key=(b['block_file'],j*cs,L*cs);on=j%L==0
                require(not on or key in origmap[L], 'Original row missing')
                r={k:b[k] for k in ['block_file','dataset','band','label','model','state','bw_label','group_id']}
                r.update(sample_id=f'L{L}h_{len(output[L]):08d}',start=j*cs,n_samples=L*cs,split=gs[gid]['split'],**dict(zip(FLAGS,flags)),keep=not any(flags),on_original_grid=on,orig_sample_id=origmap[L][key]['sample_id'] if on else '')
                output[L].append(r)
        if (gi+1)%10==0:print('GROUP',gi+1,len(by),flush=True)
    checks={}; count_details=[]
    for L,rs in output.items():
        selected=[r for r in rs if r['on_original_grid']]
        require(len(selected)==len(original[L]),f'L{L} original count')
        for a,b in zip(selected,original[L]):
            require(all(a[k]==b[k] for k in b if k!='sample_id'),f'L{L} original mismatch: '+a['sample_id'])
            require(a['orig_sample_id']==b['sample_id'],'Original ID mismatch')
        for band in FS:
            checks[f'original_L{L}_{band}']={'status':'pass','rows':sum(r['band']==band for r in selected),'flags_and_keep_equal':True}
        counts=collections.Counter(r['block_file'] for r in rs); groups=collections.defaultdict(set); blocks=collections.defaultdict(set)
        for r in rs:
            cs=round(FS[r['band']]/1000);groups[r['group_id']].add(r['split']);blocks[r['block_file']].add(r['split'])
            require(r['split']==gs[r['group_id']]['split'],'Split mismatch')
            require(0<=r['start'] and r['start']+r['n_samples']<=lengths[r['block_file']] and r['start']%cs==0 and r['n_samples']==L*cs,'Boundary mismatch')
        require(all(len(s)==1 for s in list(groups.values())+list(blocks.values())),'Leakage')
        for b in bs:
            nc=cells_count[b['block_file']]; expected=max(0,(nc-L)//(L//2)+1)
            require(counts[b['block_file']]==expected,'Block count mismatch')
            starts=[r['start'] for r in rs if r['block_file']==b['block_file']]
            require(starts==[j*round(FS[b['band']]/1000) for j in range(0,nc-L+1,L//2)],'Start grid mismatch')
            count_details.append({'L':L,'block_file':b['block_file'],'cells':nc,'expected':expected,'actual':counts[b['block_file']]})
        checks[f'leakage_boundary_count_L{L}']={'status':'pass','blocks':len(bs),'rows':len(rs),'keep':sum(r['keep'] for r in rs)}
        write(O/f'index_L{L}h.csv',rs)
        require(typed(rows(O/f'index_L{L}h.csv'))==[{**r,'on_original_grid':str(r['on_original_grid'])} for r in rs],'CSV round trip')
    dump(O/'reference_features.json',refinfo)
    dump(O/'block_counts.json',count_details)
    summary=[]
    keys=sorted(set((b['band'],b['dataset'],b['label']) for b in bs))
    for L in LS:
        for band,ds,label in keys:
            for split in ['train','val','test']:
                old=[r for r in original[L] if (r['band'],r['dataset'],r['label'],r['split'])==(band,ds,label,split)]
                new=[r for r in output.get(L,original[L]) if (r['band'],r['dataset'],r['label'],r['split'])==(band,ds,label,split)]
                summary.append(dict(L=L,band=band,dataset=ds,label=label,split=split,F_count=len(old),N_count=len(new),N_keep=sum(r['keep'] for r in new),unique_keep_seconds=round(coverage(new),9)))
    write(O/'counts.csv',summary)
    bg=[]
    for gid,g in gs.items():
        if g['dataset']=='v3' and g['label']=='background':
            bg.append(dict(band=g['band'],group_id=gid,split=g['split'],L20=sum(r['group_id']==gid for r in original[20]),L50h=sum(r['group_id']==gid for r in output[50])))
    write(O/'background_group_counts.csv',bg)
    rates=[]
    for L,rs in output.items():
        for band,state in sorted(set((r['band'],r['state']) for r in rs)):
            for kind,subset in [('original',[r for r in rs if r['on_original_grid']]),('new',[r for r in rs if not r['on_original_grid']])]:
                subset=[r for r in subset if (r['band'],r['state'])==(band,state)]
                rates.append(dict(L=L,band=band,state=state,grid=kind,n=len(subset),rejected=sum(not r['keep'] for r in subset),rejected_ratio=sum(not r['keep'] for r in subset)/len(subset) if subset else None))
    write(O/'rejection_rates.csv',rates)
    after=snapshot();dump(O/'sha256_after.json',after)
    require(before==after,'Protected files changed')
    checks['protected_sha256']={'status':'pass','files':len(before),'bytes':sum(v['bytes'] for v in before.values()),'directories':PROTECTED,'added':[],'removed':[],'changed':[]}
    verification={'status':'pass','checks':checks,'per_L':{str(L):{'rows':len(rs),'keep':sum(r['keep'] for r in rs),'original_rows':sum(r['on_original_grid'] for r in rs),'new_rows':sum(not r['on_original_grid'] for r in rs)} for L,rs in output.items()},'reference_policy':'All statistics from Task F original windows only; chars/primary from original L50','counts':summary,'background_groups':bg,'rejection_rates':rates}
    dump(O/'verification.json',verification)
    report=['# 任务 N：半重叠切片索引验收','切片：L30/40/50 的步长为 15/20/25 ms；按块从 0 开始，1 ms 网格，不跨块、不补零、不尾对齐；沿用 group_split，L10/L20 继续使用 F 索引。','筛选：复制 F 的 intervals/screening；chars、primary 仅取原 L50；dense、med、meds、outlim、tested 仅取各 L 原网格窗口，再同尺筛选全部半重叠窗口；背景全保留。','检查：三个 L、两个频段原窗口逐行元数据、5 flags 与 keep 完全一致；split/组/块无泄露；全部边界、1 ms 对齐、步长及每块数量公式通过。',f'完整性：5 个既有目录共 {len(before)} 文件、{sum(v["bytes"] for v in before.values())} 字节，前后 SHA256、路径和大小完全一致；缓存 mean/max 全用 mmap_mode=r。','决策：不导入或执行 crop.py/finalize.py；由源码文本复制函数后扩展，原统计计算保留原顺序和精度；时长按块内 keep 窗口并集求和。','计数单元格：F片数 / 本任务片数 / 本任务keep片数 / keep不重复秒；L10/20 本任务栏为 F 参照，未生成新索引。','|L(ms)|band / dataset / label|train|val|test|','|---|---|---|---|---|']
    for L in LS:
        for band,ds,label in keys:
            cells=[]
            for split in ['train','val','test']:
                r=next(r for r in summary if (r['L'],r['band'],r['dataset'],r['label'],r['split'])==(L,band,ds,label,split))
                cells.append(f'{r["F_count"]}/{r["N_count"]}/{r["N_keep"]}/{r["unique_keep_seconds"]:.3f}')
            report.append(f'|{L}|{band}/{ds}/{label}|'+ '|'.join(cells)+'|')
    report.append('v3 背景逐组（组名省略 v3:，数值 L20→L50h；T=train，V=val，E=test）：')
    for band in FS:
        report.append(band+'?'+'?'.join(f'{r["group_id"][3:]}[{dict(train="T",val="V",test="E")[r["split"]]}] {r["L20"]}?{r["L50h"]}' for r in bg if r['band']==band))
    report.append('整组切不出 L50h：'+'；'.join(f'{r["band"]} {r["group_id"]}' for r in bg if r['L50h']==0)+'；半重叠不能使不足 50 ms 的块产生 50 ms 窗口。')
    report.append('筛除比例（新增半步窗口 vs 原网格窗口；按 L30/L40/L50 顺序；分子/分母=比例）：')
    for band,state in sorted(set((r['band'],r['state']) for r in rates)):
        items=[]
        for L in output:
            pairs=[next(r for r in rates if (r['band'],r['state'],r['L'],r['grid'])==(band,state,L,g)) for g in ['new','original']]
            items.append(' vs '.join(f'{r["rejected"]}/{r["n"]}={r["rejected_ratio"]:.2%}' if r['n'] else '0/0=NA' for r in pairs))
        report.append(f'{band} × {state}：'+'；'.join(items))
    report.append('重叠片段彼此不独立；评估虚警的 Clopper–Pearson 上界应只用 on_original_grid==True 子集，或另外注明相关性与统计口径。')
    report.append('详细数字：counts.csv、background_group_counts.csv、rejection_rates.csv；逐块数量与参考特征见 block_counts.json、reference_features.json；校验见 verification.json 与前后 SHA256 清单。未训练、未生成时频图、未装包、未提交。')
    require(len(report)<=60,'Report exceeds 60 lines')
    (O/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('COMPLETE',verification['per_L'],'report_lines',len(report),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        dump(O/'failure.json',{'traceback':traceback.format_exc()}); raise
