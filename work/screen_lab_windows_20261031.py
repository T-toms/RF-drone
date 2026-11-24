"""Task B: cache-only candidate window screening; raw MAT access is gallery-only."""
from pathlib import Path
import argparse, csv, json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from build_lab_overview_20261031 import inventory
P=Path(__file__).resolve().parents[1]; A=P/'work/lab_long_overview_20261031'; OUT=P/'work/lab_window_screening_20261031'; CACHE=Path('E:/drone_lab_cache_20261031')
LS=[10,15,20,50]; CS=122880; SEED=20261031

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def js(x): return json.dumps(x,ensure_ascii=False,allow_nan=False)
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def intervals(v):
    edges=np.diff(np.r_[False,v,False].astype(int)); return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
def arrays(rec): return {k:np.load(CACHE/rec/(k+'.npy'),mmap_mode='r') for k in ['cell_valid','cell_max_db','cell_mean_db']}
def bins(m,bands,cfg):
    freq=np.array(cfg['freq_mhz_centers'][m['band']]); mask=np.array(cfg['mask']); return [mask & (freq>=b['lo_MHz']) & (freq<b['hi_MHz']) for b in bands]
def source(m,s,e):
    by={int(x['ordinal']):x for x in m['sources']}; out=[]
    for o in range(s//50,(e-1)//50+1): out.append(dict(path=by[o]['path'],ordinal=o,s=int(max(s,o*50)-o*50)*CS,e=int(min(e,(o+1)*50)-o*50)*CS))
    return out

def process(rec,cfg,bands):
    m=read(CACHE/rec/'meta.json'); ar=arrays(rec); valid=ar['cell_valid']; ix=bins(m,bands,cfg); mask=np.array(cfg['mask']); union=np.logical_or.reduce(ix) if ix else np.zeros(1024,bool)
    mx=np.asarray(ar['cell_max_db'],dtype=float); mean=np.asarray(ar['cell_mean_db'],dtype=float)
    act=np.stack([(mx[:,b]>m['floor_db']+10).mean(1) for b in ix],axis=1) if ix else np.empty((2000,0))
    power=np.stack([np.power(10,(mean[:,b]-m['floor_db'])/10).mean(1) for b in ix],axis=1) if ix else np.empty((2000,0))
    outside=mask & ~union; ao=(mx[:,outside]>m['floor_db']+10).mean(1) if outside.any() else np.zeros(2000)
    rows=[]; ret=[]; primary_by_L={}; allchars={}
    for L in LS:
        spans=[(s,s+L) for lo,hi in intervals(valid) for s in range(int(lo),int(hi)-L+1,L)]
        av=np.array([act[s:e].mean(0) for s,e in spans]); pv=np.array([10*np.log10(np.maximum(power[s:e].mean(0),1e-20)) for s,e in spans]); ov=np.array([ao[s:e].mean() for s,e in spans])
        med=np.median(av,axis=0); chars=np.flatnonzero(med>=.02); primary=int(max(chars,key=lambda k:med[k]*ix[k].sum())) if len(chars) else None
        primary_by_L[L]=primary; allchars[L]=chars.tolist()
        rr=[]
        for i,(s,e) in enumerate(spans):
            r=dict(recording=rec,L=L,window_index=i,start_cell=s,end_cell=e,epoch_start=s//1000,epoch_end=(e-1)//1000,sources=js(source(m,s,e)),**{k:m[k] for k in ['code','model','state','band','bw_label']},features=js({f'sb{k+1}':dict(A=float(av[i,k]),P=float(pv[i,k])) for k in chars}),A_out=float(ov[i]),primary_subband='none' if primary is None else f'sb{primary+1}',no_characteristic=primary is None)
            rr.append(r)
        for ref in ['recording','epoch']:
            for group in ([np.arange(len(rr))] if ref=='recording' else [np.array([i for i,r in enumerate(rr) if r['epoch_start']==ep]) for ep in [0,1]]):
                if not len(group): continue
                reasons={int(i):[] for i in group}; mp=np.zeros(len(group),bool); ma=mp.copy()
                if primary is not None:
                    medg=np.median(av[group],axis=0); mp=av[group,primary]<.25*medg[primary]; ma=np.all(av[group][:,chars]<.25*medg[chars],axis=1)
                    values=[(f'sb{k+1}_act',np.log1p(av[:,k]),.05) for k in chars]+[(f'sb{k+1}_pow',pv[:,k],1.) for k in chars]+[('out_act',np.log1p(ov),.05)]
                    for name,v,fl in values:
                        center=np.median(v[group]); scale=max(1.4826*np.median(abs(v[group]-center)),fl); z=(v[group]-center)/scale
                        for j in np.flatnonzero(abs(z)>3.5): reasons[int(group[j])].append(name+('_high' if z[j]>0 else '_low'))
                for j,i in enumerate(group):
                    rr[i].update({f'missing_primary_{ref}':bool(mp[j]),f'missing_all_{ref}':bool(ma[j]),f'differ_{ref}':bool(reasons[int(i)]),f'differ_reason_{ref}':js(reasons[int(i)]),f'discard_{ref}':bool(mp[j] or reasons[int(i)])})
        rows+=rr
    # A single 1-ms primary sequence, anchored to the shortest window, independent of L.
    gp=primary_by_L[10]; gaps=[]
    if gp is not None:
        for lo,hi in intervals(valid): gaps += [int(e-s) for s,e in intervals(act[lo:hi,gp]==0)]
    gap=[float(np.percentile(gaps,q)) if gaps else 0. for q in [50,95,100]]
    for L in LS:
        p=primary_by_L[L]; subset=[r for r in rows if r['L']==L]
        for ref in ['recording','epoch']:
            counts={f'n_{k}':sum(r[f'{k}_{ref}'] for r in subset) for k in ['missing_primary','missing_all','differ','discard']}
            ret.append(dict(recording=rec,L=L,ref=ref,band=m['band'],state=m['state'],n_windows=len(subset),**counts,retention=1-counts['n_discard']/len(subset),no_characteristic=p is None,primary_subband='none' if p is None else f'sb{p+1}',primary_mhz=js(None if p is None else bands[p]),gap_primary_subband='none' if gp is None else f'sb{gp+1}',gap_p50_ms=gap[0],gap_p95_ms=gap[1],gap_max_ms=gap[2]))
    recommendation={}
    for ref in ['recording','epoch']: recommendation[ref]=next((str(r['L']) for r in ret if r['ref']==ref and not r['no_characteristic'] and r['retention']>=.9),'none')
    fig,axes=plt.subplots(5,1,figsize=(16,9),gridspec_kw={'height_ratios':[6,1,1,1,1]},sharex=True,layout='constrained')
    fc=m['fc_hz']/1e6; cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#bdbdbd')
    im=axes[0].imshow((mx-m['floor_db']).T,origin='lower',aspect='auto',extent=(0,2000,fc-61.44,fc+61.44),vmin=-5,vmax=40,cmap=cmap,interpolation='nearest'); axes[0].set_ylabel('Frequency (MHz)'); fig.colorbar(im,ax=axes[0],label='dB above floor')
    if gp is not None: axes[0].add_patch(Rectangle((0,bands[gp]['lo_MHz']),2000,bands[gp]['hi_MHz']-bands[gp]['lo_MHz'],fill=False,edgecolor='red',linewidth=.8))
    for ax,L in zip(axes[1:],LS):
        ax.set_facecolor('#bdbdbd'); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_ylabel(str(L)+' ms')
        for r in rows:
            if r['L']!=L: continue
            color='#bdbdbd' if r['no_characteristic'] else ('#d62728' if r['missing_primary_recording'] else '#ff9900' if r['differ_recording'] else '#2ca02c')
            ax.add_patch(Rectangle((r['start_cell'],0),L,1,color=color,linewidth=0))
    for ax in axes: ax.axvline(1000,color='white',ls='--'); ax.set_xlim(0,2000)
    axes[-1].set_xlabel('Time in recording (ms)')
    fig.suptitle(rec+' | recommended recording='+recommendation['recording']+'; epoch='+recommendation['epoch'])
    fig.savefig(OUT/(rec+'_screening.png'),dpi=110); plt.close(fig)
    recommendation['recording_id']=rec; recommendation['recommended_recording']=recommendation.pop('recording'); recommendation['recommended_epoch']=recommendation.pop('epoch')
    return rows,ret,recommendation

def summary(rows,ret,recs):
    lines=['# 窗口筛选候选结果','规则为候选标记，是否采用由用户决定；未训练、未划分、未更改原数据。','FS 假设 122.88 MS/s；按有效连续区间非重叠切窗。主子带间隙使用 L=10 所选主子带的 1 ms 序列，所有 L 共用。']
    def rate(rs): return f"{1-sum(r['n_discard'] for r in rs)/sum(r['n_windows'] for r in rs):.2%}"
    for L in LS:
        for ref in ['recording','epoch']:
            rs=[r for r in ret if r['L']==L and r['ref']==ref and not r['no_characteristic']]
            groups='；'.join(k+'='+rate([r for r in rs if r[field]==k]) for field in ['band','state'] for k in sorted({r[field] for r in rs}))
            lines.append(f'- {L} ms / {ref}：总体 {rate(rs)}；'+groups)
    for ref in ['recording','epoch']: lines.append('推荐窗长 '+ref+'：'+js(dict(Counter(r['recommended_'+ref] for r in recs))))
    for L in [10,20]:
        lines.append(f'保留率最低 10 个（{L} ms，recording）：')
        for r in sorted([r for r in ret if r['L']==L and r['ref']=='recording' and not r['no_characteristic']],key=lambda r:r['retention'])[:10]:
            rr=[x for x in rows if x['recording']==r['recording'] and x['L']==L]; counts=Counter(z for x in rr for z in json.loads(x['differ_reason_recording'])); counts['missing_primary']=r['n_missing_primary']
            lines.append(f"- {r['recording']}：{r['retention']:.2%}；{counts.most_common(3)}")
    dif=[]
    for i in range(0,len(ret),2):
        x,y=ret[i:i+2]
        if abs(x['retention']-y['retention'])>.1+1e-12: dif.append(f"{x['recording']}({x['L']}ms,{x['retention']-y['retention']:+.3f})")
    lines.append('两种参照差 >0.10：'+('；'.join(dif) or '无'))
    lines.append('无特征录制/窗长：'+('；'.join(sorted({r['recording']+f"/{r['L']}ms" for r in ret if r['no_characteristic']})) or '无'))
    lines.append('间隙 p95≥窗长（可能空窗）：'+ '；'.join(f"{r['recording']}/{r['L']}ms(p95={r['gap_p95_ms']:g})" for r in ret if r['ref']=='recording' and r['gap_p95_ms']>=r['L']))
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def verify():
    with (OUT/'window_index.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    rng=np.random.default_rng(SEED); result={}; errors=[]; invalid=0; nchecks=0
    for d in [CACHE/rec for rec in sorted(read(A/'occupied_bands.json'))]:
        ar=arrays(d.name); v=ar['cell_valid']; rr=[r for r in rows if r['recording']==d.name]
        for L in LS:
            # independent run lengths, without intervals helper
            runs=[]; n=0
            for ok in list(v)+[False]:
                if ok: n+=1
                elif n: runs.append(n); n=0
            expected=sum(x//L for x in runs); actual=sum(int(r['L'])==L for r in rr); nchecks+=1
            if actual!=expected: errors.append([d.name,L,expected,actual])
        invalid+=sum(not v[int(r['start_cell']):int(r['end_cell'])].all() for r in rr)
    result['1_window_counts']=dict(status='pass' if not errors else 'fail',checks=nchecks,errors=errors)
    result['2_valid_cells']=dict(status='pass' if not invalid else 'fail',invalid_windows=invalid)
    cfg=read(A/'config.json'); bands=read(A/'occupied_bands.json'); samples=[]
    eligible=[r for r in rows if r['primary_subband']!='none']
    for i in rng.choice(len(eligible),5,replace=False):
        r=eligible[i]; m=read(CACHE/r['recording']/'meta.json'); ar=arrays(r['recording']); k=int(r['primary_subband'][2:])-1; b=bins(m,bands[r['recording']],cfg)[k]; s,e=int(r['start_cell']),int(r['end_cell']); av=float((ar['cell_max_db'][s:e][:,b].astype(float)>m['floor_db']+10).mean()); pv=float(10*np.log10(np.power(10,(ar['cell_mean_db'][s:e][:,b].astype(float)-m['floor_db'])/10).mean())); f=json.loads(r['features'])[r['primary_subband']]; samples.append(dict(recording=r['recording'],start=s,A_error=abs(av-f['A']),P_error=abs(pv-f['P'])))
    result['3_feature_recompute']=dict(status='pass' if all(x['A_error']<1e-4 and x['P_error']<.01 for x in samples) else 'fail',samples=samples)
    ss=[]
    for i in rng.choice(len(rows),3,replace=False):
        r=rows[i]; src=json.loads(r['sources']); spans=[(x['ordinal']*50+x['s']//CS,x['ordinal']*50+x['e']//CS) for x in src]; ok=spans[0][0]==int(r['start_cell']) and spans[-1][1]==int(r['end_cell']) and all(spans[j][1]==spans[j+1][0] for j in range(len(spans)-1)) and sum(x['e']-x['s'] for x in src)==int(r['L'])*CS; ss.append(dict(recording=r['recording'],start=r['start_cell'],pass_check=ok))
    cross=[r for r in rows if len(json.loads(r['sources']))>1]; badcross=sum(sum(x['e']-x['s'] for x in json.loads(r['sources']))!=int(r['L'])*CS for r in cross)
    result['4_sources']=dict(status='pass' if all(x['pass_check'] for x in ss) and not badcross else 'fail',samples=ss,cross_segment_windows=len(cross),cross_length_errors=badcross)
    pngs=[OUT/(d.name+'_screening.png') for d in [CACHE/rec for rec in sorted(read(A/'occupied_bands.json'))]]+list(OUT.glob('gallery_*.png')); sizes={p.name:p.stat().st_size if p.exists() else 0 for p in pngs}; ng=len(list(OUT.glob('gallery_*.png')))
    result['5_pngs']=dict(status='pass' if len(pngs)==104 and ng==8 and min(sizes.values())>30*1024 else 'fail',count=len(pngs),gallery_count=ng,sizes=sizes)
    after=inventory(); before=read(OUT/'source_inventory_before.json'); dump(OUT/'source_inventory_after.json',after); result['6_source_unchanged']=dict(status='pass' if before==after else 'fail',before=before,after=after); dump(OUT/'verification.json',result)
    print(js({k:v['status'] for k,v in result.items()}),flush=True)
    assert all(x['status']=='pass' for x in result.values())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--only',nargs='+'); ap.add_argument('--verify',action='store_true'); args=ap.parse_args()
    if args.verify: verify(); return
    OUT.mkdir(exist_ok=True); before=inventory()
    if not (OUT/'source_inventory_before.json').exists(): dump(OUT/'source_inventory_before.json',before)
    cfg=read(A/'config.json'); bands=read(A/'occupied_bands.json'); dump(OUT/'config.json',dict(window_ms=LS,cell_samples=CS,fs_assumed_hz=122880000,n_cells=2000,threshold_db=10,seed=SEED,characteristic_min=.02,missing_fraction=.25,z_threshold=3.5,mad_scale=1.4826,z_floors=[.05,1.,.05],minimum_retention=.9,gap_primary='L=10 recording primary; independent 1-ms sequence',rules_original=(P/'work/codex_tasks/TASK_B_窗口筛选_20261031.md').read_text(encoding='utf-8'),cache_config=cfg))
    rows=[]; ret=[]; recs=[]
    for rec in sorted(bands):
        if args.only and rec.split('__')[0] not in args.only: continue
        x,y,z=process(rec,cfg,bands[rec]); rows+=x; ret+=y; recs.append(z); print(rec,flush=True)
    if not args.only:
        csvwrite(OUT/'window_index.csv',rows); csvwrite(OUT/'recording_retention.csv',ret); csvwrite(OUT/'recommended_window.csv',recs); summary(rows,ret,recs)
if __name__=='__main__': main()
