"""Task C cache-only screening. Raw MAT is read only by the gallery."""
from screen_lab_windows_20261031 import *
OUT=P/'work/lab_window_screening_v2_20261031'
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
    outside=mask & ~np.logical_or.reduce(ix); ao=(mx[:,outside]>m['floor_db']+THRESH_DB).mean(1) if outside.any() else np.zeros(len(valid))
    floor=np.percentile(mean[:,mask],10,axis=1); data={}
    for L in LS:
        spans=[(s,s+L) for lo,hi in intervals(valid) for s in range(int(lo),int(hi)-L+1,L)]
        data[L]=(spans,np.array([act[s:e].mean(0) for s,e in spans]),np.array([10*np.log10(power[s:e].mean(0)) for s,e in spans]),np.array([ao[s:e].mean() for s,e in spans]))
    refs={}
    for ref in ['recording','epoch']:
        for ep in ([-1] if ref=='recording' else [0,1]):
            spans,av,pv,ov=data[50]; g=np.array([i for i,(s,e) in enumerate(spans) if ep==-1 or s//1000==ep],int)
            occ=(av[g]>=APPEAR).mean(0) if len(g) else np.zeros(len(ix)); chars=np.flatnonzero(occ>=PERSIST).tolist()
            primary=int(max(chars,key=lambda k:np.median(av[g,k])*ix[k].sum())) if chars else int(np.argmax(occ))
            refs[ref,ep]=(chars,primary)
    rows=[]; ret=[]
    for L,(spans,av,pv,ov) in data.items():
        rr=[]
        for i,(s,e) in enumerate(spans):
            fw=float(np.median(floor[s:e])); rr.append(dict(recording=rec,L=L,window_index=i,start_cell=s,end_cell=e,epoch_start=s//1000,epoch_end=(e-1)//1000,sources=js(source(m,s,e)),**{k:m[k] for k in ['code','model','state','band','bw_label']},features=js({f'sb{k+1}':dict(A=float(av[i,k]),P=float(pv[i,k])) for k in range(len(ix))}),A_out=float(ov[i]),F_win=fw,dF=fw-m['floor_db'],primary_subband=f'sb{refs["recording",-1][1]+1}',no_characteristic=False))
        for ref in ['recording','epoch']:
            for ep in ([-1] if ref=='recording' else [0,1]):
                g=np.array([i for i,(s,e) in enumerate(spans) if ep==-1 or s//1000==ep],int)
                if not len(g): continue
                chars,primary=refs[ref,ep]; dense=[k for k in chars if (av[g,k]>=APPEAR).mean()>=PERSIST]; med=np.median(av[g],0); medp=np.median(pv[g],0); outlim=max(OUT_MIN,OUT_MULT*np.median(ov[g]))
                for i in g:
                    lack=[k for k in chars if av[i,k]<APPEAR]; shifts=[f'sb{k+1}_'+('high' if pv[i,k]>medp[k] else 'low') for k in dense if abs(pv[i,k]-medp[k])>POWER_DB]
                    flags=[bool(lack),bool(av[i,primary]<WEAK*med[primary]),bool(shifts),rr[i]['dF']>FLOOR_DB,bool(ov[i]>outlim)]
                    rr[i].update({f'{k}_{ref}':v for k,v in zip(FLAGS,flags)}); rr[i].update({f'discard_{ref}':any(flags),f'reasons_{ref}':';'.join(k for k,v in zip(FLAGS,flags) if v),f'lack_which_{ref}':js([bands[k] for k in lack]),f'power_shift_which_{ref}':';'.join(shifts),f'persistent_{ref}':js(chars),f'dense_{ref}':js(dense),f'primary_subband_{ref}':f'sb{primary+1}',f'no_persistent_{ref}':not bool(chars),f'A_primary_{ref}':float(av[i,primary])})
            subset=rr; counts={f'n_{k}':sum(r[f'{k}_{ref}'] for r in subset) for k in FLAGS+['discard']}
            primary=refs['recording',-1][1]; gaps=[]
            for lo,hi in intervals(valid):
                for s,e in intervals(act[lo:hi,primary]==0): gaps.append(int(e-s))
            gap=[float(np.percentile(gaps,q)) if gaps else 0. for q in [50,95,100]]
            ret.append(dict(recording=rec,L=L,ref=ref,band=m['band'],state=m['state'],n_windows=len(rr),**counts,retention=1-counts['n_discard']/len(rr),feature_coverage=1-counts['n_lack_feature']/len(rr),no_characteristic=False,no_persistent=any(r[f'no_persistent_{ref}'] for r in rr),primary_subband=f'sb{primary+1}',primary_mhz=js(bands[primary]),gap_p50_ms=gap[0],gap_p95_ms=gap[1],gap_max_ms=gap[2]))
        rows+=rr
    recommendation={f'recommended_{ref}':next((str(r['L']) for r in ret if r['ref']==ref and r['retention']>=MIN_RET),'none') for ref in ['recording','epoch']}; recommendation['recording_id']=rec
    gp=refs['recording',-1][1]; fig,axes=plt.subplots(5,1,figsize=(16,9),gridspec_kw={'height_ratios':[6,1,1,1,1]},sharex=True,layout='constrained'); fc=m['fc_hz']/1e6
    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#bdbdbd'); image=np.ma.array((mx-m['floor_db']).T,mask=np.broadcast_to(~valid,(1024,len(valid))))
    im=axes[0].imshow(image,origin='lower',aspect='auto',extent=(0,2000,fc-61.44,fc+61.44),vmin=-5,vmax=40,cmap=cmap,interpolation='nearest'); fig.colorbar(im,ax=axes[0],label='dB above floor'); axes[0].set_ylabel('MHz')
    for k in refs['recording',-1][0]: axes[0].add_patch(Rectangle((0,bands[k]['lo_MHz']),2000,bands[k]['hi_MHz']-bands[k]['lo_MHz'],fill=False,edgecolor='white',linewidth=.6))
    axes[0].add_patch(Rectangle((0,bands[gp]['lo_MHz']),2000,bands[gp]['hi_MHz']-bands[gp]['lo_MHz'],fill=False,edgecolor='red',linewidth=1))
    for ax,L in zip(axes[1:],LS):
        ax.set_facecolor('#bdbdbd'); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_ylabel(f'{L} ms')
        for r in rows:
            if r['L']!=L: continue
            c='#d62728' if r['lack_feature_recording'] or r['primary_weak_recording'] else '#9467bd' if r['floor_raised_recording'] else '#ff9900' if r['discard_recording'] else '#2ca02c'; ax.add_patch(Rectangle((r['start_cell'],0),L,1,color=c,linewidth=0))
    for ax in axes: ax.axvline(1000,color='white',ls='--'); ax.set_xlim(0,2000)
    axes[-1].set_xlabel('Time (ms)'); fig.suptitle(rec+' | '+js(recommendation)); fig.savefig(OUT/(rec+'_screening.png'),dpi=110); plt.close(fig)
    return rows,ret,recommendation

def summary(rows,ret,recs):
    lines=['# 任务 C：窗口筛选 v2','缓存筛选；仅画廊读取 MAT；未训练、未划分、未修改原始数据。','epoch 跨界窗归 epoch_start；无持续特征时按 50 ms 出现率回退；灰色仅表示无效 cell。']
    def rates(rs):
        n=sum(r['n_windows'] for r in rs); return f"保留 {1-sum(r['n_discard'] for r in rs)/n:.2%} / 覆盖 {1-sum(r['n_lack_feature'] for r in rs)/n:.2%}"
    for L in LS:
        for ref in ['recording','epoch']:
            rs=[r for r in ret if r['L']==L and r['ref']==ref]; lines.append(f'{L}ms/{ref}：'+rates(rs)+'；'+'；'.join(k+':'+rates([r for r in rs if r[f]==k]) for f in ['band','state'] for k in sorted({r[f] for r in rs})))
    for L in [10,20]:
        for ref in ['recording','epoch']:
            rr=[r for r in rows if r['L']==L]; lines.append(f'触发占比 {L}ms/{ref}：'+'；'.join(f'{k}={sum(r[k+"_"+ref] for r in rr)/len(rr):.2%}' for k in FLAGS))
        for r in sorted([r for r in ret if r['L']==L and r['ref']=='recording'],key=lambda r:r['retention'])[:5]: lines.append(f"最低 {L}ms：{r['recording']} {r['retention']:.2%}；主因 "+max(FLAGS,key=lambda k:r['n_'+k]))
    for ref in ['recording','epoch']: lines.append('推荐 '+ref+'：'+js(dict(Counter(r['recommended_'+ref] for r in recs))))
    npers=sorted({r['recording'] for r in ret if r['no_persistent']}); csvwrite(OUT/'no_persistent.csv',[dict(recording=r) for r in npers] or [dict(recording='none')]); lines.append('no_persistent 清单：'+('无' if not npers else '见 no_persistent.csv（'+str(len(npers))+' 个）'))
    with (BASE/'recording_retention.csv').open(encoding='utf-8') as f: old=list(csv.DictReader(f))
    for L in LS:
        for ref in ['recording','epoch']:
            a=[r for r in old if int(r['L'])==L and r['ref']==ref]; b=[r for r in ret if r['L']==L and r['ref']==ref]; x=1-sum(int(r['n_discard']) for r in a)/sum(int(r['n_windows']) for r in a); y=1-sum(r['n_discard'] for r in b)/sum(r['n_windows'] for r in b); lines.append(f'对比 B {L}ms/{ref}：{x:.2%} → {y:.2%}（{(y-x)*100:+.2f} 百分点）')
    lines.append('间隙为合并主子带的有效 1ms 序列；所有 L/ref 共用 recording 主子带间隙，逐窗参照主子带完整记录在索引。'); assert len(lines)<=50; (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def verify():
    with (OUT/'window_index.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    with (BASE/'window_index.csv').open(encoding='utf-8') as f: old=list(csv.DictReader(f))
    result={}; counts=lambda rs:Counter((r['recording'],r['L']) for r in rs)
    result['1_counts']=dict(status='pass' if counts(rows)==counts(old) else 'fail',groups=len(counts(rows)))
    cfg=read(A/'config.json'); bands=read(OUT/'merged_bands.json'); target=[b for r,bs in bands.items() if r.startswith('007_5.8GHz_40M-signal') for b in bs if b['lo_MHz']<=5778 and b['hi_MHz']>=5813]; result['2_dc_merge']=dict(status='pass' if target else 'fail',bands=target)
    samples=[]; rng=np.random.default_rng(SEED)
    for i in rng.choice(len(rows),5,replace=False):
        r=rows[i]; m=read(CACHE/r['recording']/'meta.json'); ar=arrays(r['recording']); s,e=int(r['start_cell']),int(r['end_cell']); ix=bins(m,bands[r['recording']],cfg); fs=json.loads(r['features']); errs=[]
        for k,b in enumerate(ix):
            ac=float((ar['cell_max_db'][s:e][:,b].astype(float)>m['floor_db']+THRESH_DB).mean()); pw=float(10*np.log10(np.power(10,(ar['cell_mean_db'][s:e][:,b].astype(float)-m['floor_db'])/10).mean())); errs.append([abs(ac-fs[f'sb{k+1}']['A']),abs(pw-fs[f'sb{k+1}']['P'])])
        df=float(np.median(np.percentile(ar['cell_mean_db'][s:e][:,np.array(cfg['mask'])].astype(float),10,axis=1))-m['floor_db']); samples.append(dict(recording=r['recording'],start=s,A_error=max(x[0] for x in errs),P_error=max(x[1] for x in errs),dF_error=abs(df-float(r['dF']))))
    result['3_recompute']=dict(status='pass' if all(x['A_error']<1e-4 and x['P_error']<.01 and x['dF_error']<.01 for x in samples) else 'fail',samples=samples)
    sets={}
    for r in rows:
        for ref in ['recording','epoch']:
            key=(r['recording'],ref,r['epoch_start'] if ref=='epoch' else '-1'); sets.setdefault(key,{}).setdefault(r['L'],set()).add(r['persistent_'+ref])
    bad=[str(k) for k,v in sets.items() if len(set.union(*v.values()))!=1 or len(v)!=4]; result['4_persistent']=dict(status='pass' if not bad else 'fail',errors=bad)
    rr=[r for r in rows if r['recording']=='002_2.4GHz_10M-signal__2026-09-20-19-24-45' and r['L']=='10' and int(r['start_cell'])<=1050 and int(r['end_cell'])>=1060]; result['5_raised_floor']=dict(status='withdrawn',note='coordinator expectation withdrawn: independent gallery color scales made a weak-signal window appear brighter; retain original 3 dB threshold',windows=[dict(start=r['start_cell'],dF=r['dF']) for r in rr])
    pngs=[OUT/(r+'_screening.png') for r in bands]+[OUT/('gallery_'+p['recording']+'.png') for p in read(OUT/'gallery_selection.json')]; sizes={p.name:p.stat().st_size if p.exists() else 0 for p in pngs}; result['6_pngs']=dict(status='pass' if len(pngs)==104 and min(sizes.values())>30*1024 else 'fail',sizes=sizes)
    after=inventory(); dump(OUT/'source_inventory_after.json',after); result['6_source_unchanged']=dict(status='pass' if after==read(OUT/'source_inventory_before.json') else 'fail',after=after)
    dump(OUT/'verification.json',result); print(js({k:v['status'] for k,v in result.items()})); assert all(v['status'] in ['pass','withdrawn'] for v in result.values())

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
    OUT.mkdir(exist_ok=True); dump(OUT/'source_inventory_before.json',inventory()); cfg=read(A/'config.json'); original=read(A/'occupied_bands.json'); bands={r:merge_bands(read(CACHE/r/'meta.json'),bs,cfg) for r,bs in original.items()}; dump(OUT/'merged_bands.json',bands)
    dump(OUT/'config.json',dict(window_ms=LS,appear=APPEAR,persistent=PERSIST,dense=PERSIST,weak=WEAK,power_db=POWER_DB,floor_db=FLOOR_DB,out_min=OUT_MIN,out_multiplier=OUT_MULT,threshold_db=THRESH_DB,minimum_retention=MIN_RET,seed=SEED,cell_samples=CS,dc_bins=[511,512,513],merge_other_bins=2,cache_config=cfg,rules=(P/'work/codex_tasks/TASK_C_筛选规则修订_20261031.md').read_text(encoding='utf-8')))
    rows=[]; ret=[]; recs=[]
    for rec,bs in bands.items():
        x,y,z=process(rec,cfg,bs); rows+=x; ret+=y; recs.append(z); print(rec,flush=True)
    csvwrite(OUT/'window_index.csv',rows); csvwrite(OUT/'recording_retention.csv',ret); csvwrite(OUT/'recommended_window.csv',recs); summary(rows,ret,recs); finalize()
if __name__=='__main__': main()
