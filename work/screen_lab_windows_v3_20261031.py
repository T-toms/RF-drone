"""Task D cache-only screening. Raw MAT is read only by the gallery."""
from screen_lab_windows_20261031 import *
OUT=P/'work/lab_window_screening_v3_20261031'
COUT=P/'work/lab_window_screening_v2_20261031'
BASE=P/'work/lab_window_screening_20261031'
APPEAR=.01; PERSIST=.90; WEAK=.25; POWER_DB=6.; FLOOR_DB=3.; OUT_MIN=.05; OUT_MULT=3.; THRESH_DB=10.; MIN_RET=.90
FLAGS=['lack_feature','primary_weak','power_shift','floor_raised','interference']

def merge_bands(m,bs,cfg):
    freq=np.array(cfg['freq_mhz_centers'][m['band']]); merged=[]
    for b in sorted(bs,key=lambda x:x['lo_MHz']):
        b=dict(b)
        if merged:
            gap=np.flatnonzero((freq>=merged[-1]['hi_MHz']-1e-8)&(freq<b['lo_MHz']-1e-8))
            if sum(int(i) not in [511,512,513] for i in gap)<=2:
                merged[-1]['hi_MHz']=b['hi_MHz']; merged[-1]['width_MHz']=b['hi_MHz']-merged[-1]['lo_MHz']; continue
        merged.append(b)
    return merged

def process(rec,cfg,bands):
    m=read(CACHE/rec/'meta.json'); ar=arrays(rec); valid=ar['cell_valid']; ix=bins(m,bands,cfg); mask=np.array(cfg['mask']); mx=np.asarray(ar['cell_max_db'],float); mean=np.asarray(ar['cell_mean_db'],float)
    act=np.stack([(mx[:,b]>m['floor_db']+THRESH_DB).mean(1) for b in ix],1)
    power=np.stack([np.power(10,(mean[:,b]-m['floor_db'])/10).mean(1) for b in ix],1)
    outside=mask & ~np.logical_or.reduce(ix) & ~environment_mask(m,cfg); ao=(mx[:,outside]>m['floor_db']+THRESH_DB).mean(1) if outside.any() else np.zeros(len(valid))
    floor=np.percentile(mean[:,mask],10,axis=1); data={}
    for L in LS:
        spans=[(s,s+L) for lo,hi in intervals(valid) for s in range(int(lo),int(hi)-L+1,L)]
        data[L]=(spans,np.array([act[s:e].mean(0) for s,e in spans]),np.array([10*np.log10(power[s:e].mean(0)) for s,e in spans]),np.array([ao[s:e].mean() for s,e in spans]))
    sv={}
    for L,(spans,_,_,_) in data.items():
        sv[L]=np.array([[signal_strength(mx[s:e,b],m["floor_db"]) for b in ix] for s,e in spans])
    eligible=[k for k,b in enumerate(bands) if not b["environment"]]
    refs={}
    for ref in ['recording','epoch']:
        for ep in ([-1] if ref=='recording' else [0,1]):
            spans,av,pv,ov=data[50]; g=np.array([i for i,(s,e) in enumerate(spans) if ep==-1 or s//1000==ep],int)
            occ=(av[g]>=APPEAR).mean(0) if len(g) else np.zeros(len(ix)); chars=[k for k in eligible if occ[k]>=PERSIST]
            primary=int(max(chars,key=lambda k:np.median(av[g,k])*ix[k].sum())) if chars else (max(eligible,key=lambda k:occ[k]) if eligible else -1)
            refs[ref,ep]=(chars,primary)
    rows=[]; ret=[]
    for L,(spans,av,pv,ov) in data.items():
        rr=[]
        for i,(s,e) in enumerate(spans):
            fw=float(np.median(floor[s:e])); rr.append(dict(recording=rec,L=L,window_index=i,start_cell=s,end_cell=e,epoch_start=s//1000,epoch_end=(e-1)//1000,sources=js(source(m,s,e)),**{k:m[k] for k in ['code','model','state','band','bw_label']},features=js({f'sb{k+1}':dict(A=float(av[i,k]),P=float(pv[i,k]),S=None if np.isnan(sv[L][i,k]) else float(sv[L][i,k])) for k in range(len(ix))}),A_out=float(ov[i]),F_win=fw,dF=fw-m['floor_db'],primary_subband=f'sb{refs["recording",-1][1]+1}' if refs['recording',-1][1]>=0 else 'none',no_characteristic=not bool(eligible)))
        for ref in ['recording','epoch']:
            for ep in ([-1] if ref=='recording' else [0,1]):
                g=np.array([i for i,(s,e) in enumerate(spans) if ep==-1 or s//1000==ep],int)
                if not len(g): continue
                chars,primary=refs[ref,ep]; dense=[k for k in chars if (av[g,k]>=APPEAR).mean()>=PERSIST]; med=np.median(av[g],0); meds=np.array([np.median(sv[L][g,k][np.isfinite(sv[L][g,k])]) if np.isfinite(sv[L][g,k]).any() else np.nan for k in range(len(ix))]); outlim=max(OUT_MIN,OUT_MULT*np.median(ov[g]))
                for i in g:
                    lack=[k for k in dense if av[i,k]<APPEAR]; tested=set(([primary] if primary>=0 else [])+[k for k in dense if med[k]>=.20]); shifts=[f'sb{k+1}_'+('high' if sv[L][i,k]>meds[k] else 'low') for k in sorted(tested) if abs(sv[L][i,k]-meds[k])>POWER_DB]
                    flags=[bool(lack),bool(primary>=0 and av[i,primary]<WEAK*med[primary]),bool(shifts),rr[i]['dF']>FLOOR_DB,bool(ov[i]>outlim)]
                    rr[i].update({f'{k}_{ref}':v for k,v in zip(FLAGS,flags)}); rr[i].update({f'discard_{ref}':any(flags),f'reasons_{ref}':';'.join(k for k,v in zip(FLAGS,flags) if v),f'lack_which_{ref}':js([bands[k] for k in lack]),f'power_shift_which_{ref}':';'.join(shifts),f'persistent_{ref}':js(chars),f'dense_{ref}':js(dense),f'primary_subband_{ref}':f'sb{primary+1}' if primary>=0 else 'none',f'no_persistent_{ref}':not bool(chars),f'A_primary_{ref}':float(av[i,primary]) if primary>=0 else float('nan'),f'S_primary_{ref}':float(sv[L][i,primary]) if primary>=0 else float('nan'),f'coverage_{ref}':float(np.mean(av[i,chars]>=APPEAR)) if chars else 1.})
            subset=rr; counts={f'n_{k}':sum(r[f'{k}_{ref}'] for r in subset) for k in FLAGS+['discard']}
            primary=refs['recording',-1][1]; gaps=[]
            for lo,hi in intervals(valid):
                for s,e in intervals(act[lo:hi,primary]==0) if primary>=0 else []: gaps.append(int(e-s))
            gap=[float(np.percentile(gaps,q)) if gaps else 0. for q in [50,95,100]]
            ret.append(dict(recording=rec,L=L,ref=ref,band=m['band'],state=m['state'],n_windows=len(rr),**counts,retention=1-counts['n_discard']/len(rr),feature_coverage=float(np.mean([r['coverage_'+ref] for r in rr])),mean_coverage=float(np.mean([r['coverage_'+ref] for r in rr])),full_coverage_rate=float(np.mean([r['coverage_'+ref]==1 for r in rr])),no_characteristic=not bool(eligible),no_persistent=any(r[f'no_persistent_{ref}'] for r in rr),primary_subband=f'sb{primary+1}' if primary>=0 else 'none',primary_mhz=js(bands[primary] if primary>=0 else None),gap_p50_ms=gap[0],gap_p95_ms=gap[1],gap_max_ms=gap[2]))
        rows+=rr
    recommendation={'recording_id':rec}
    for ref in ['recording','epoch']:
        keep=next((r['L'] for r in ret if r['ref']==ref and r['retention']>=MIN_RET),None)
        cover=next((r['L'] for r in ret if r['ref']==ref and r['full_coverage_rate']>=MIN_RET),None)
        recommendation['rec_keep_'+ref]=str(keep) if keep else 'none'
        recommendation['rec_cover_'+ref]=str(cover) if cover else 'none'
        recommendation['recommended_'+ref]=str(max(keep,cover)) if keep and cover else 'none'
    gp=refs['recording',-1][1]; fig,axes=plt.subplots(5,1,figsize=(16,9),gridspec_kw={'height_ratios':[6,1,1,1,1]},sharex=True,layout='constrained'); fc=m['fc_hz']/1e6
    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#bdbdbd'); image=np.ma.array((mx-m['floor_db']).T,mask=np.broadcast_to(~valid,(1024,len(valid))))
    im=axes[0].imshow(image,origin='lower',aspect='auto',extent=(0,2000,fc-61.44,fc+61.44),vmin=-5,vmax=40,cmap=cmap,interpolation='nearest'); fig.colorbar(im,ax=axes[0],label='dB above floor'); axes[0].set_ylabel('MHz')
    for band in read(OUT/'environment_bands.json')[m['band']]:
        axes[0].add_patch(Rectangle((0,band['lo_MHz']),2000,band['hi_MHz']-band['lo_MHz'],fill=False,edgecolor='cyan',linestyle='--',linewidth=1))
    for k in refs['recording',-1][0]: axes[0].add_patch(Rectangle((0,bands[k]['lo_MHz']),2000,bands[k]['hi_MHz']-bands[k]['lo_MHz'],fill=False,edgecolor='white',linewidth=.6))
    if gp>=0: axes[0].add_patch(Rectangle((0,bands[gp]['lo_MHz']),2000,bands[gp]['hi_MHz']-bands[gp]['lo_MHz'],fill=False,edgecolor='red',linewidth=1))
    for ax,L in zip(axes[1:],LS):
        ax.set_facecolor('#bdbdbd'); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_ylabel(f'{L} ms')
        for r in rows:
            if r['L']!=L: continue
            c='#d62728' if r['lack_feature_recording'] or r['primary_weak_recording'] else '#9467bd' if r['floor_raised_recording'] else '#ff9900' if r['discard_recording'] else '#2ca02c'; ax.add_patch(Rectangle((r['start_cell'],0),L,1,color=c,linewidth=0))
    for ax in axes: ax.axvline(1000,color='white',ls='--'); ax.set_xlim(0,2000)
    axes[-1].set_xlabel('Time (ms)'); fig.suptitle(rec+' | '+js(recommendation)); fig.savefig(OUT/(rec+'_screening.png'),dpi=110); plt.close(fig)
    return rows,ret,recommendation

def summary(rows,ret,recs):
    lines=['# 任务 D：窗口筛选 v3','缓存筛选；仅画廊读取 MAT；未训练、未划分、未修改原始数据。','epoch 跨界窗归 epoch_start；无持续特征时按 50 ms 出现率回退；灰色仅表示无效 cell。']
    def rates(rs):
        n=sum(r['n_windows'] for r in rs); return f"保留 {1-sum(r['n_discard'] for r in rs)/n:.2%} / mean_coverage {sum(r['mean_coverage']*r['n_windows'] for r in rs)/n:.2%} / full_coverage_rate {sum(r['full_coverage_rate']*r['n_windows'] for r in rs)/n:.2%}"
    for L in LS:
        for ref in ['recording','epoch']:
            rs=[r for r in ret if r['L']==L and r['ref']==ref]; lines.append(f'{L}ms/{ref}：'+rates(rs)+'；'+'；'.join(k+':'+rates([r for r in rs if r[f]==k]) for f in ['band','state'] for k in sorted({r[f] for r in rs})))
    for L in [10,20]:
        for ref in ['recording','epoch']:
            rr=[r for r in rows if r['L']==L]; lines.append(f'触发占比 {L}ms/{ref}：'+'；'.join(f'{k}={sum(r[k+"_"+ref] for r in rr)/len(rr):.2%}' for k in FLAGS))
        for r in sorted([r for r in ret if r['L']==L and r['ref']=='recording'],key=lambda r:r['retention'])[:5]: lines.append(f"最低 {L}ms：{r['recording']} {r['retention']:.2%}；主因 "+max(FLAGS,key=lambda k:r['n_'+k]))
    for ref in ['recording','epoch']:
        for kind in ['rec_keep','rec_cover','recommended']: lines.append(kind+' '+ref+'：'+js(dict(Counter(r[kind+'_'+ref] for r in recs))))
    npers=sorted({r['recording'] for r in ret if r['no_persistent']}); csvwrite(OUT/'no_persistent.csv',[dict(recording=r) for r in npers] or [dict(recording='none')]); lines.append('no_persistent 清单：'+('无' if not npers else '见 no_persistent.csv（'+str(len(npers))+' 个）'))
    with (COUT/'recording_retention.csv').open(encoding='utf-8') as f: old=list(csv.DictReader(f))
    for L in LS:
        for ref in ['recording','epoch']:
            a=[r for r in old if int(r['L'])==L and r['ref']==ref]; b=[r for r in ret if r['L']==L and r['ref']==ref]; x=1-sum(int(r['n_discard']) for r in a)/sum(int(r['n_windows']) for r in a); y=1-sum(r['n_discard'] for r in b)/sum(r['n_windows'] for r in b); lines.append(f'对比 C {L}ms/{ref}：{x:.2%} → {y:.2%}（{(y-x)*100:+.2f} 百分点）')
    lines.append('环境区间（n_recordings为区间内bin最低覆盖数，all_bins_recordings为整段共同覆盖数）：'+js(read(OUT/'environment_bands.json')))
    lines.append('协调者修订：环境频点为同频段≥80%录制 occ_any≥0.05 的掩膜内bin；子带≥50% bin为环境才整体排除，其余剔除环境bin；非稠密持续特征见 non_dense_persistent.csv。')
    lines.append('无持续特征时 coverage=1；主子带从非环境子带按50ms出现率回退，无非环境子带时主子带为none且主信号/功率标记为假。')
    lines.append('间隙为合并主子带的有效 1ms 序列；所有 L/ref 共用 recording 主子带间隙，逐窗参照主子带完整记录在索引。'); assert len(lines)<=50; (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def finalize():
    with (OUT/'window_index.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    with (OUT/'recording_retention.csv').open(encoding='utf-8') as f: ret=list(csv.DictReader(f))
    for rec in sorted({r['recording'] for r in ret}):
        anchor=next(r for r in ret if r['recording']==rec and r['ref']=='recording' and r['L']=='10')
        for r in ret:
            if r['recording']==rec:
                for k in ['primary_subband','primary_mhz','gap_p50_ms','gap_p95_ms','gap_max_ms']: r[k]=anchor[k]
    csvwrite(OUT/'recording_retention.csv',ret)
    stats=[]
    for rec in sorted({r['recording'] for r in rows}):
        for ref in ['recording','epoch']:
            rr=[r for r in rows if r['recording']==rec]
            for ep in (['all'] if ref=='recording' else ['0','1']):
                for L in LS:
                    group=[r for r in rr if r['L']==str(L) and (ep=='all' or r['epoch_start']==ep)]
                    if not group: continue
                    a=group[0]; stats.append(dict(recording=rec,ref=ref,epoch=ep,L=L,n_windows=len(group),persistent=a['persistent_'+ref],dense=a['dense_'+ref],primary_subband=a['primary_subband_'+ref],no_persistent=a['no_persistent_'+ref]))
    csvwrite(OUT/'reference_features.csv',stats)

def main():
    import sys
    if '--verify' in sys.argv: verify(); return
    if '--finalize' in sys.argv: finalize(); return
    OUT.mkdir(exist_ok=True); dump(OUT/'source_metadata_before.json',[dict(path=str(p.relative_to('G:/drone')).replace(chr(92),'/'),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in Path('G:/drone').rglob('*') if p.is_file()]); dump(OUT/'source_inventory_before.json',inventory()); cfg=read(A/'config.json'); original=read(A/'occupied_bands.json'); bands={r:merge_bands(read(CACHE/r/'meta.json'),bs,cfg) for r,bs in original.items()}; detect_environment(bands); dump(OUT/'merged_bands.json',bands)
    dump(OUT/'config.json',dict(window_ms=LS,appear=APPEAR,persistent=PERSIST,dense=PERSIST,weak=WEAK,power_db=POWER_DB,floor_db=FLOOR_DB,out_min=OUT_MIN,out_multiplier=OUT_MULT,threshold_db=THRESH_DB,minimum_retention=MIN_RET,environment_rule="Coordinator clarification: masked folded bins occ_any>=0.05 in >=80% recordings; whole subband environmental if >=50% bins, otherwise strip environmental bins",seed=SEED,cell_samples=CS,dc_bins=[511,512,513],merge_other_bins=2,cache_config=cfg,rules=(P/'work/codex_tasks/TASK_D_筛选规则v3_20261031.md').read_text(encoding='utf-8')))
    rows=[]; ret=[]; recs=[]
    for rec,bs in bands.items():
        x,y,z=process(rec,cfg,bs); rows+=x; ret+=y; recs.append(z); print(rec,flush=True)
    csvwrite(OUT/'window_index.csv',rows); csvwrite(OUT/'recording_retention.csv',ret); csvwrite(OUT/'recommended_window.csv',recs); summary(rows,ret,recs); finalize()



def signal_strength(block,floor):
    active=(block>floor+THRESH_DB).any(axis=1)
    return float(np.median(block[active].mean(axis=1)-floor)) if active.sum()>=2 else np.nan

_original_bins=bins

def environment_mask(m,cfg):
    freq=np.array(cfg['freq_mhz_centers'][m['band']]); result=np.zeros(len(freq),bool)
    for band in read(OUT/'environment_bands.json')[m['band']]: result |= (freq>=band['lo_MHz']-1e-8)&(freq<band['hi_MHz']-1e-8)
    return result

def bins(m,bands,cfg):
    ix=_original_bins(m,bands,cfg); env=environment_mask(m,cfg)
    return [b if band['environment'] else b & ~env for b,band in zip(ix,bands)]

def detect_environment(bands):
    cfg=read(A/'config.json'); mask=np.array(cfg['mask']); groups={}
    for rec in bands:
        band=read(CACHE/rec/'meta.json')['band']; groups.setdefault(band,[]).append(rec)
    env={}
    for band,recs in groups.items():
        occ=np.array([np.load(CACHE/r/'occ_any.npy')>=.05 for r in recs]); counts=occ.sum(axis=0); selected=mask & (counts>=.8*len(recs)); freq=np.array(cfg['freq_mhz_centers'][band]); step=float(np.median(np.diff(freq))); entries=[]
        for lo,hi in intervals(selected):
            entries.append(dict(lo_MHz=float(freq[lo]-step*3/8),hi_MHz=float(freq[hi-1]+step*5/8),n_recordings=int(counts[lo:hi].min()),all_bins_recordings=int(np.sum(np.all(occ[:,lo:hi],axis=1))),min_bin_recordings=int(counts[lo:hi].min()),total_recordings=len(recs)))
        env[band]=entries
        for rec in recs:
            for b in bands[rec]:
                ix=mask & (freq>=b['lo_MHz']) & (freq<b['hi_MHz']); b['environment']=bool(np.mean(selected[ix])>=.5)
    dump(OUT/'environment_bands.json',env)

def verify():
    def load(path):
        with path.open(encoding='utf-8') as f: return list(csv.DictReader(f))
    rows=load(OUT/'window_index.csv'); old=load(BASE/'window_index.csv'); bands=read(OUT/'merged_bands.json'); cfg=read(A/'config.json'); env=read(OUT/'environment_bands.json'); result={}
    key=lambda r:tuple(r[k] for k in ['recording','L','window_index','start_cell','end_cell','sources'])
    result['1_windows']=dict(status='pass' if [key(r) for r in rows]==[key(r) for r in old] else 'fail',count=len(rows))
    covered=[b for band,bs in env.items() if '2.4' in band for b in bs if b['lo_MHz']<=2426.9 and b['hi_MHz']>=2427.-1e-8]
    result['2_environment']=dict(status='pass' if covered else 'fail',bands=covered)
    target='012_2.4GHz_10M-signal__2026-09-20-18-58-55'; rr=[r for r in rows if r['recording']==target and r['L']=='10']; primary=bands[target][int(rr[0]['primary_subband_recording'][2:])-1]; rate=np.mean([float(r['A_primary_recording'])>=APPEAR for r in rr])
    result['3_primary']=dict(status='pass' if rate>=.95 and primary['lo_MHz']<=2401.5 and primary['hi_MHz']>=2410.5 else 'fail',appearance_rate=float(rate),band=primary)
    samples=[]
    for i in np.random.default_rng(SEED).choice(len(rows),5,replace=False):
        r=rows[i]; m=read(CACHE/r['recording']/'meta.json'); ar=arrays(r['recording']); ix=bins(m,bands[r['recording']],cfg); s,e=int(r['start_cell']),int(r['end_cell']); checks=[]
        # Independent calculation directly from cached cells, no production S helper.
        for ref in ['recording','epoch']:
            if r['primary_subband_'+ref]=='none':
                checks.append(np.isnan(float(r['S_primary_'+ref])) and np.isnan(float(r['A_primary_'+ref])) and float(r['coverage_'+ref])==1.)
                continue
            k=int(r['primary_subband_'+ref][2:])-1; block=np.array(ar['cell_max_db'][s:e][:,ix[k]],float); active=np.max(block,axis=1)>m['floor_db']+10; S=float(np.median(np.sum(block[active],axis=1)/block.shape[1])-m['floor_db']) if sum(active)>=2 else float('nan'); avalue=float(np.count_nonzero(block>m['floor_db']+10)/block.size)
            chars=json.loads(r['persistent_'+ref]); present=[np.count_nonzero(ar['cell_max_db'][s:e][:,ix[j]].astype(float)>m['floor_db']+10)/( (e-s)*sum(ix[j]))>=.01 for j in chars]; coverage=sum(present)/len(chars) if chars else 1.
            stored=float(r['S_primary_'+ref]); checks.append(bool((np.isnan(S) and np.isnan(stored) or abs(S-stored)<1e-8) and abs(avalue-float(r['A_primary_'+ref]))<1e-8 and abs(coverage-float(r['coverage_'+ref]))<1e-8))
        samples.append(dict(recording=r['recording'],start=s,pass_check=all(checks)))
    result['4_recompute']=dict(status='pass' if all(x['pass_check'] for x in samples) else 'fail',samples=samples)
    bad=sum((r['discard_'+ref]=='True')!=any(r[k+'_'+ref]=='True' for k in FLAGS) for r in rows for ref in ['recording','epoch']); result['5_flags']=dict(status='pass' if not bad else 'fail',errors=bad)
    pngs=[OUT/(rec+'_screening.png') for rec in bands]+[OUT/('gallery_'+p['recording']+'.png') for p in read(OUT/'gallery_selection.json')]; sizes={p.name:p.stat().st_size if p.exists() else 0 for p in pngs}; result['6_pngs']=dict(status='pass' if len(pngs)==104 and min(sizes.values())>30*1024 else 'fail',sizes=sizes)
    after=inventory(); dump(OUT/'source_inventory_after.json',after); result['6_source_unchanged']=dict(status='pass' if after==read(OUT/'source_inventory_before.json') else 'fail')
    full=[dict(path=str(p.relative_to('G:/drone')).replace(chr(92),'/'),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in Path('G:/drone').rglob('*') if p.is_file()]
    dump(OUT/'source_metadata_after.json',full)
    result['6_source_metadata']=dict(status='pass' if full==read(OUT/'source_metadata_before.json') else 'fail',files=len(full),comparison='relative path, size, mtime_ns; baseline before gallery')
    dump(OUT/'verification.json',result); print(js({k:v['status'] for k,v in result.items()})); assert all(v['status']=='pass' for v in result.values())

def non_dense_report():
    with (OUT/'window_index.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    bands=read(OUT/'merged_bands.json'); records=[]
    for rec in bands:
        for ref in ['recording','epoch']:
            for ep in (['all'] if ref=='recording' else ['0','1']):
                rr=[r for r in rows if r['recording']==rec and (ep=='all' or r['epoch_start']==ep)]
                chars=json.loads(rr[0]['persistent_'+ref])
                for k in chars:
                    rates={str(L):np.mean([json.loads(r['features'])[f'sb{k+1}']['A']>=APPEAR for r in rr if r['L']==str(L)]) for L in [10,15,20]}
                    if any(v<PERSIST for v in rates.values()): records.append(dict(recording=rec,ref=ref,epoch=ep,subband=f'sb{k+1}',lo_MHz=bands[rec][k]['lo_MHz'],hi_MHz=bands[rec][k]['hi_MHz'],appearance_10=rates['10'],appearance_15=rates['15'],appearance_20=rates['20']))
    csvwrite(OUT/'non_dense_persistent.csv',records)

if __name__=='__main__':
    main()
    import sys
    if '--verify' not in sys.argv: non_dense_report()
