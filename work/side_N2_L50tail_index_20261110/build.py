"""Task N2: copied Task N screening; original L50 windows plus one block-tail window.
Run with the workspace .venv Python -I; no source-module import or execution.
"""
import csv, json, hashlib, collections, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
O = Path(__file__).resolve().parent
F = ROOT/'work/crop_slice_20261107'
LS = [50]
FLAGS = ['lack_feature','primary_weak','power_shift','floor_raised','interference']
FS = {'2.4GHz':61440000,'5.8GHz':100000000}
FC = {'2.4GHz':2430,'5.8GHz':5800}
WIDTH = {'2.4GHz':25,'5.8GHz':45}
PROTECTED = ['crop_slice_20261107','train_stage1_L20_20261108','side_N_overlap_index_20261110','side_L_wifimix_20261110','side_M_notchfill_20261110']

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
  if L != 50: continue
  ss=[];offset=0
  for b,a in zip(gb,ar):
   ss.extend((offset+j,offset+j+L,b,j) for j in tail_starts(len(a['mean'])))
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

def tail_starts(n):
    starts=list(range(0,n-50+1,50))
    if starts and starts[-1]+50<n: starts.append(n-50)
    return starts

def main():
    before=snapshot(); dump(O/'sha256_before.json',before)
    print('HASH_BEFORE',len(before),sum(v['bytes'] for v in before.values()),flush=True)
    bs=json.loads((F/'plan.json').read_text(encoding='utf-8-sig'))
    blockrows={r['block_file']:r for r in rows(F/'blocks.csv')}
    gs={r['group_id']:r for r in rows(F/'group_split.csv')}
    original=typed(rows(F/'index_L50.csv')); l20=typed(rows(F/'index_L20.csv'))
    origmap={(r['block_file'],r['start'],r['n_samples']):r for r in original}
    lengths={}; cells_count={}; by=collections.defaultdict(list)
    for b in bs: by[b['group_id']].append(b)
    output=[]; refinfo=[]
    for gi,(gid,gb) in enumerate(by.items()):
        ars=[]
        for b in gb:
            cell=F/'cells'/Path(b['block_file']).stem
            ar={k:np.load(cell/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['mean','max']}
            require(all(not a.flags.writeable for a in ar.values()),'Cache not read only')
            z=np.load(b['block_file'],mmap_mode='r',allow_pickle=False)
            n=len(z); del z
            require(n==int(blockrows[b['block_file']]['n_samples']),'Block length differs from blocks.csv')
            require(len(ar['mean'])==len(ar['max'])==n//round(FS[b['band']]/1000),'Cell count differs from block length')
            lengths[b['block_file']]=n; cells_count[b['block_file']]=len(ar['mean']); ars.append(ar)
        if gb[0]['label']=='drone':
            rr,info=screening(gb,ars); refinfo.append(info); rr=rr[50]
        else:
            rr=[(b,j,[False]*5) for b,a in zip(gb,ars) for j in tail_starts(len(a['mean']))]
        for b,j,flags in rr:
            cs=round(FS[b['band']]/1000); key=(b['block_file'],j*cs,50*cs); on=j%50==0
            require(not on or key in origmap,'Original row missing')
            r={k:b[k] for k in ['block_file','dataset','band','label','model','state','bw_label','group_id']}
            r.update(sample_id=f'L50t_{len(output):08d}',start=j*cs,n_samples=50*cs,split=gs[gid]['split'],**dict(zip(FLAGS,flags)),keep=not any(flags),on_original_grid=on,orig_sample_id=origmap[key]['sample_id'] if on else '')
            output.append(r)
        if (gi+1)%10==0: print('GROUP',gi+1,len(by),flush=True)
    checks={}; selected=[r for r in output if r['on_original_grid']]
    require(len(selected)==len(original),'Original count mismatch')
    for a,b in zip(selected,original):
        require(all(a[k]==b[k] for k in b if k!='sample_id'),'Original metadata/flags/keep mismatch: '+a['sample_id'])
        require(a['orig_sample_id']==b['sample_id'],'Original ID mismatch')
    checks['original_rows']={'status':'pass','rows':len(selected),'metadata_equal':True,'five_flags_equal':True,'keep_equal':True,'orig_sample_id_equal':True}
    grouped=collections.defaultdict(list); groups=collections.defaultdict(set); blocks=collections.defaultdict(set)
    for r in output:
        cs=round(FS[r['band']]/1000); grouped[r['block_file']].append(r)
        groups[r['group_id']].add(r['split']); blocks[r['block_file']].add(r['split'])
        require(r['split']==gs[r['group_id']]['split'],'Split mismatch')
        require(0<=r['start'] and r['start']+r['n_samples']<=lengths[r['block_file']] and r['start']%cs==0 and r['n_samples']==50*cs,'Boundary mismatch')
        require(r['keep']==(not any(r[k] for k in FLAGS)),'Keep differs from flags')
        require(r['label']!='background' or (r['keep'] and not any(r[k] for k in FLAGS)),'Background screened')
    require(all(len(s)==1 for s in list(groups.values())+list(blocks.values())),'Leakage')
    details=[]; overlaps=collections.defaultdict(list)
    for b in bs:
        bf=b['block_file']; n=cells_count[bf]; cs=round(FS[b['band']]/1000); rs=grouped[bf]
        expected=list(range(0,n-50+1,50))
        if n>=50 and n%50: expected.append(n-50)
        require([r['start'] for r in rs]==[j*cs for j in expected],'Start grid/count mismatch')
        tails=[r for r in rs if not r['on_original_grid']]
        require(len(tails)<=1,'Multiple tail windows')
        require(len(tails)==int(n>=50 and n%50!=0),'Tail presence mismatch')
        for r in tails:
            require(r['start']+r['n_samples']==n*cs,'Tail end mismatch')
            require(r['orig_sample_id']=='','Tail original ID must be empty')
            overlap=(n//50)*50-(n-50)
            require(0<overlap<50,'Tail overlap invalid')
            overlaps[b['dataset']].append(overlap)
        details.append(dict(block_file=bf,cells=n,block_samples=lengths[bf],expected=len(expected),actual=len(rs),tail_count=len(tails)))
    require(len({r['sample_id'] for r in output})==len(output),'Duplicate sample ID')
    require(len({(r['block_file'],r['start']) for r in output})==len(output),'Duplicate window')
    checks['split_boundary_tail']={'status':'pass','blocks':len(bs),'groups':len(groups),'background_unscreened':True,'tail_end_is_full_cells':True,'unique_ids_and_windows':True}
    write(O/'index_L50t.csv',output)
    require(typed(rows(O/'index_L50t.csv'))==[{**r,'on_original_grid':str(r['on_original_grid'])} for r in output],'CSV round trip')
    checks['csv_round_trip']={'status':'pass'}
    dump(O/'reference_features.json',refinfo); dump(O/'block_counts.json',details)
    summary=[]
    keys=sorted(set((b['band'],b['dataset'],b['label']) for b in bs))
    for band,ds,label in keys:
        for split in ['train','val','test']:
            key=(band,ds,label,split)
            sub=lambda rr:[r for r in rr if (r['band'],r['dataset'],r['label'],r['split'])==key]
            old=sub(original); new=sub(output); ref=sub(l20)
            summary.append(dict(band=band,dataset=ds,label=label,split=split,F_L50_count=len(old),L50t_count=len(new),L50t_keep=sum(r['keep'] for r in new),L50t_keep_unique_seconds=round(coverage(new),9),L20_keep_unique_seconds=round(coverage(ref),9)))
    write(O/'counts.csv',summary)
    bg=[]
    for gid,g in gs.items():
        if g['dataset']=='v3' and g['label']=='background':
            bg.append(dict(band=g['band'],group_id=gid,split=g['split'],L20=sum(r['group_id']==gid for r in l20),L50t=sum(r['group_id']==gid for r in output)))
    write(O/'background_group_counts.csv',bg)
    overlap_stats=[]
    for ds in sorted(set(b['dataset'] for b in bs)):
        vals=overlaps[ds]
        overlap_stats.append(dict(dataset=ds,n=len(vals),min_ms=min(vals) if vals else None,median_ms=float(np.median(vals)) if vals else None,max_ms=max(vals) if vals else None,distribution_ms=dict(sorted(collections.Counter(vals).items()))))
    dump(O/'tail_overlap.json',overlap_stats)
    rates=[]
    for ds in ['ALL']+sorted(set(b['dataset'] for b in bs)):
        for kind,on in [('original',True),('tail',False)]:
            sub=[r for r in output if r['on_original_grid']==on and (ds=='ALL' or r['dataset']==ds)]
            rates.append(dict(dataset=ds,grid=kind,n=len(sub),rejected=sum(not r['keep'] for r in sub),rejected_ratio=sum(not r['keep'] for r in sub)/len(sub) if sub else None))
    write(O/'rejection_rates.csv',rates)
    after=snapshot(); dump(O/'sha256_after.json',after)
    require(before==after,'Protected files changed')
    checks['protected_sha256']={'status':'pass','files':len(before),'bytes':sum(v['bytes'] for v in before.values()),'directories':PROTECTED,'added':[],'removed':[],'changed':[]}
    verification={'status':'pass','checks':checks,'rows':len(output),'keep':sum(r['keep'] for r in output),'original_rows':len(selected),'tail_rows':len(output)-len(selected),'reference_policy':'Task N screening copied; only original Task F L50 windows determine window-reference statistics; all full cells determine spectral floor/subbands as in Task N; background unscreened','counts':summary,'background_groups':bg,'tail_overlap':overlap_stats,'rejection_rates':rates}
    dump(O/'verification.json',verification)
    report=['# 任务 N2：L50 原网格加块尾补窗（仅索引）',
        '规则：每块原窗口 j=0,50,…；n≥50 且 n%50≠0 时追加 j=n−50；n<50 不切片。L10–L40 继续用 F，未生成或修改其索引。',
        '筛选：文本复制任务 N 的 screening()/参考统计代码到本目录，仅将待筛窗口改为 L50 原网格加尾窗；窗口参考量只取 F 原 L50 网格，背景不筛选。',
        '检查：原窗口按行元数据、5 flags、keep、orig_sample_id 全一致；split 与 group_split 一致且组/块无泄露；边界、尾窗数、尾端对齐、逐块数量、唯一性、CSV 回读通过。',
        f'保护：五个指定目录共 {len(before)} 文件、{sum(v["bytes"] for v in before.values())} 字节，前后 SHA256、路径和大小完全一致；cells mean/max 均只读 mmap。',
        '决策：沿用 N 的全组完整 cells 计算频谱底噪/子带；窗口参考量仅原 L50；不导入或执行 crop.py/finalize.py。块尾按完整 1 ms cells 结束（不使用不足 1 ms 余样）。',
        '计数：各格为 F L50片数 / L50t片数 / L50t keep片数 / L50t keep不重复秒 / L20 keep不重复秒；时长为逐块 keep 窗口并集。',
        '|band / dataset / label|train|val|test|','|---|---|---|---|']
    for band,ds,label in keys:
        cells=[]
        for split in ['train','val','test']:
            r=next(r for r in summary if (r['band'],r['dataset'],r['label'],r['split'])==(band,ds,label,split))
            cells.append(f'{r["F_L50_count"]}/{r["L50t_count"]}/{r["L50t_keep"]}/{r["L50t_keep_unique_seconds"]:.3f}/{r["L20_keep_unique_seconds"]:.3f}')
        report.append(f'|{band}/{ds}/{label}|'+ '|'.join(cells)+'|')
    report.append('v3 背景逐组片数（省略 v3:；T=train，V=val，E=test；L20→L50t）：')
    for band in FS:
        report.append(band+'：'+'；'.join(f'{r["group_id"][3:]}[{dict(train="T",val="V",test="E")[r["split"]]}] {r["L20"]}→{r["L50t"]}' for r in bg if r['band']==band))
    report.append('块尾与前一原窗口重叠分布（单位 ms；dataset：尾窗数，最小/中位/最大）：')
    for r in overlap_stats: report.append(f'{r["dataset"]}：{r["n"]}，{r["min_ms"]}/{r["median_ms"]}/{r["max_ms"]}；完整频数见 tail_overlap.json。')
    report.append('筛除比例（尾窗 vs 原窗口；拒绝数/总数=比例）：')
    for ds in ['ALL']+sorted(set(b['dataset'] for b in bs)):
        rr=[next(r for r in rates if r['dataset']==ds and r['grid']==g) for g in ['tail','original']]
        report.append(ds+'：'+' vs '.join(f'{r["rejected"]}/{r["n"]}={r["rejected_ratio"]:.2%}' if r['n'] else '0/0=NA' for r in rr))
    report.append(f'总计 {len(output)} 片（原 {len(selected)}，尾 {len(output)-len(selected)}），keep {sum(r["keep"] for r in output)}；新增列 on_original_grid、orig_sample_id，sample_id 前缀 L50t_。')
    report.append('明细见 counts.csv、background_group_counts.csv、rejection_rates.csv、block_counts.json、reference_features.json；验收见 verification.json 及两份 SHA256 清单。仅用标准库与 numpy；未训练、未生成时频图、未装包、未提交。')
    require(len(report)<=40,'Report exceeds 40 lines')
    (O/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('COMPLETE',verification['rows'],verification['keep'],'tail',verification['tail_rows'],'report_lines',len(report),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        dump(O/'failure.json',{'traceback':traceback.format_exc()}); raise