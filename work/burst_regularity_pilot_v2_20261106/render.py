from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pilot import *
from matplotlib.collections import PatchCollection

def render():
    results=json.loads((OUT/(sys.argv[1] if len(sys.argv)>1 else 'results.json')).read_text(encoding='utf8'));alls=[];spans=[];checks=[]
    bg=results[-1]['signals']
    for r in results:
        rec=r['recording'];idx=r['index'];df=pd.read_csv(OUT/f'bursts_{rec}.csv');ss=r['signals'];mapping={s['signal_id']:s for s in ss}
        for s in ss:
            s['also_in_background']=idx!=8 and any(abs(s['bandwidth_MHz']/b['bandwidth_MHz']-1)<.2 and abs(s['duration_ms']/b['duration_ms']-1)<.2 and any(abs(c['center']-d['center'])<1 for c in s['channel_metrics'].values() for d in b['channel_metrics'].values()) for b in bg)
            alls.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in s.items()})
        featured=df[df.signal_id.isin([s['signal_id'] for s in ss if s['classification']=='featured'])]
        bands=intervals_union(list(zip(featured.f_lo_MHz,featured.f_hi_MHz)))
        spans.append(dict(recording=rec,featured_span_MHz=(bands[-1][1]-bands[0][0]) if bands else 0,featured_lo_MHz=bands[0][0] if bands else None,featured_hi_MHz=bands[-1][1] if bands else None,featured_total_bw_MHz=sum(b-a for a,b in bands),featured_plus_review_span_MHz=(lambda x: float(x.f_hi_MHz.max()-x.f_lo_MHz.min()) if len(x) else 0)(df[df.signal_id.isin([s['signal_id'] for s in ss if s['classification'] in ['featured','review']])]),fragment_count=int(df.fragment.sum()),featured_subbands=json.dumps(bands),**{k:sum(s['classification']==k for s in ss) for k in ['featured','review','isolated']}))
        ov=np.load(OUT/f'overview_{idx}.npz');active=[s for s in ss if s['classification']!='isolated'];colors={'featured':'lime','review':'orange','isolated':'red'}
        fig,ax=plt.subplots(2,1,figsize=(18,10),gridspec_kw={'height_ratios':[3,1]});extent=[0,r['duration_ms'],ov['freq'][0],ov['freq'][-1]]
        display=np.full((max(1,int(np.ceil(r['duration_ms']))),len(ov['freq'])),np.nan,dtype='float32'); positions=np.rint(ov['times']).astype(int); good=positions<len(display);display[positions[good]]=ov['power'][good]-r['floor_db']
        ax[0].imshow(display.T,origin='lower',aspect='auto',extent=extent,vmin=-5,vmax=35,cmap='magma')
        for category in colors:
            sub=df[df.signal_id.isin([s['signal_id'] for s in ss if s['classification']==category])];rects=[Rectangle((e.t_start_ms,e.f_lo_MHz),e.duration_ms,e.bandwidth_MHz) for e in sub.itertuples()];ax[0].add_collection(PatchCollection(rects,facecolor='none',edgecolor=colors[category],linewidth=.35,alpha=.6))
        for s in active: ax[1].hist(s['iat_ms'],bins=np.linspace(0,200,101),histtype='step',label=f"S{s['signal_id']} {s['classification']} {s['rule']} T0={s['T0_ms']:.3f}")
        ax[0].set(title=rec+' | independent v3 blocks concatenated for display',xlabel='Time (ms)',ylabel='Frequency (MHz)');ax[1].set(xlabel='IAT (ms)',ylabel='Count');
        if active:ax[1].legend(fontsize=5,ncol=4)
        fig.tight_layout();fig.savefig(OUT/f'annot_{rec}.png',dpi=130);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6));sc=ax.scatter(df.bandwidth_MHz,df.duration_ms,c=df.signal_id,cmap='turbo',s=3);ax.scatter(df[df.fragment].bandwidth_MHz,df[df.fragment].duration_ms,c='gray',s=1,label='fragment');ax.set(xscale='log',yscale='log',xlabel='Bandwidth (MHz)',ylabel='Duration (ms)',title=rec+' | DBSCAN shape clusters');fig.tight_layout();fig.savefig(OUT/f'clusters_{rec}.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(18,12))
        for ax,v in zip(axes,r['views']):
            z=np.load(OUT/f"view_{idx}_{v['chunk']}.npz");start=float(z['start']);end=start+len(z['p'])*float(z['dt']);sel=df[(df.t_start_ms<end)&(df.t_end_ms>start)];ax.imshow((z['p']-z['floor']).T,origin='lower',aspect='auto',extent=[start,end,z['freq'][0],z['freq'][-1]],vmin=-5,vmax=35,cmap='magma')
            for sid,sub in sel.groupby('signal_id'):
                color='gray' if sid<0 else plt.cm.tab20(int(sid)%20)
                rects=[Rectangle((e.t_start_ms,e.f_lo_MHz),e.duration_ms,e.bandwidth_MHz) for e in sub.itertuples()]
                ax.add_collection(PatchCollection(rects,facecolor='none',edgecolor=color,linewidth=.3 if sid<0 else .7))
                if sid>=0:
                    for e in sub.itertuples():ax.text(e.t_start_ms,e.f_hi_MHz,f'S{int(sid)}',color=color,fontsize=4,clip_on=True)
            ntrue=int((~sel.fragment).sum())
            ax.set(xlim=(start,end),xlabel='Time (ms)',ylabel='Frequency (MHz)',title=f"Random chunk {v['chunk']}: {ntrue} merged nonfragment events")
            checks.append(dict(recording=rec,chunk=v['chunk'],start_ms=start,end_ms=end,n_intersecting_events=ntrue,n_fragments=int(sel.fragment.sum()),manual_count=None,relative_difference=None,count_agreement_pass=False,visual_check='pending'))
        fig.suptitle(rec+' | raw STFT, signal colored boxes; gray = fragment');fig.tight_layout();fig.savefig(OUT/f'zoom_{rec}.png',dpi=120);plt.close(fig)
        print('RENDER',rec, 'signals',len(ss),flush=True)
    if len(results)<8:return
    pd.DataFrame(alls).to_csv(OUT/'signals.csv',index=False);pd.DataFrame(spans).to_csv(OUT/'spans.csv',index=False)
    pngs=list(OUT.glob('*.png'));dump(OUT/'verification.json',dict(random_seed=20261106,random_views=checks,cross_boundary_examples=[dict(recording=r['recording'],**b) for r in results for b in r['boundaries']][:20],source_before=json.loads((OUT/'source_before.json').read_text()),source_after=json.loads((OUT/'source_after.json').read_text()),source_unchanged=(OUT/'source_before.json').read_text()==(OUT/'source_after.json').read_text(),png_sizes={p.name:p.stat().st_size for p in pngs},pngs_complete=all((OUT/f'{prefix}_{rec}.png').exists() and (OUT/f'{prefix}_{rec}.png').stat().st_size>30000 for prefix in ['annot','zoom','clusters'] for rec in RECS)))

if __name__=='__main__':render()
