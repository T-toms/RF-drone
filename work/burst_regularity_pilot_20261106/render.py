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
        spans.append(dict(recording=rec,featured_span_MHz=(bands[-1][1]-bands[0][0]) if bands else 0,featured_lo_MHz=bands[0][0] if bands else None,featured_hi_MHz=bands[-1][1] if bands else None,featured_total_bw_MHz=sum(b-a for a,b in bands),featured_subbands=json.dumps(bands),**{k:sum(s['classification']==k for s in ss) for k in ['featured','review','isolated']}))
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
        fig,ax=plt.subplots(figsize=(10,6));sc=ax.scatter(df.bandwidth_MHz,df.duration_ms,c=df.signal_id,cmap='turbo',s=2);ax.set(xscale='log',yscale='log',xlabel='Bandwidth (MHz)',ylabel='Duration (ms)',title=rec+' | DBSCAN shape clusters');fig.tight_layout();fig.savefig(OUT/f'clusters_{rec}.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(18,12))
        for ax,v in zip(axes,r['views']):
            z=np.load(OUT/f"view_{idx}_{v['chunk']}.npz");start=float(z['start']);end=start+len(z['p'])*float(z['dt']);sel=df[(df.t_start_ms<end)&(df.t_end_ms>start)];ax.imshow((z['p']-z['floor']).T,origin='lower',aspect='auto',extent=[start,end,z['freq'][0],z['freq'][-1]],vmin=-5,vmax=35,cmap='magma')
            rects=[Rectangle((e.t_start_ms,e.f_lo_MHz),e.duration_ms,e.bandwidth_MHz) for e in sel.itertuples()];ax.add_collection(PatchCollection(rects,facecolor='none',edgecolor='cyan',linewidth=.35));ax.set(xlim=(start,end),xlabel='Time (ms)',ylabel='Frequency (MHz)',title=f"Random chunk {v['chunk']}: {len(sel)} intersecting events")
            checks.append(dict(recording=rec,chunk=v['chunk'],start_ms=start,end_ms=end,n_intersecting_events=len(sel),visual_check='pending'))
        fig.suptitle(rec+' | raw STFT, detected boxes in cyan');fig.tight_layout();fig.savefig(OUT/f'zoom_{rec}.png',dpi=120);plt.close(fig)
        print('RENDER',rec, 'signals',len(ss),flush=True)
    if len(results)<8:return
    pd.DataFrame(alls).to_csv(OUT/'signals.csv',index=False);pd.DataFrame(spans).to_csv(OUT/'spans.csv',index=False)
    pngs=list(OUT.glob('*.png'));dump(OUT/'verification.json',dict(random_seed=20261106,random_views=checks,cross_boundary_examples=[dict(recording=r['recording'],**b) for r in results for b in r['boundaries']][:20],source_before=json.loads((OUT/'source_before.json').read_text()),source_after=json.loads((OUT/'source_after.json').read_text()),source_unchanged=(OUT/'source_before.json').read_text()==(OUT/'source_after.json').read_text(),png_sizes={p.name:p.stat().st_size for p in pngs},pngs_complete=len(pngs)==24 and all(p.stat().st_size>30000 for p in pngs)))
    lines=['# 信号规律性分析试点（2026-11-06）','仅分析指定 8 个录制；未截频、切片、训练或划分，原始数据只读。','参数：1024 点 Hann，hop=512；归一化功率 /sum(window²)，阈值 floor+10 dB；3×2 闭运算、8 连通、面积≥6。','细线屏蔽 ±150 kHz；>20 ms 连通域按时间占用率<50%切开。DBSCAN：标准化 log带宽、log时长、平均功率，eps=0.45、min_samples=5；以0.05标准差网格加权聚类降低内存，属于近似；噪声点各自独立。','周期：IAT 与≤200 ms两两差直方图（0.05 ms量化自相关计算）候选，以±3%T0整数倍匹配率选周期；不跨 v3 NN 块算间隔。','分类按任务 a/b/c 常量（见 pilot.py RULES）；50 ms出现率按突发与窗口相交计数；100 ms覆盖独立计算。','v3不同NN拼接仅为显示，epoch代表独立块；实际总时长约500 ms，不虚构2000 ms观测。','全部信号的频率、带宽、时长、T0、信道/间隔、每epoch/信道指标及分类见 signals.csv；每突发见 bursts_*.csv。']
    for r,sp in zip(results,spans):
        lines.append(f"## #{r['index']} {r['recording']}")
        lines.append(f"featured/review/isolated={sp['featured']}/{sp['review']}/{sp['isolated']}；featured span={sp['featured_span_MHz']:.3f} MHz，总带宽={sp['featured_total_bw_MHz']:.3f} MHz。")
        top=sorted(r['signals'],key=lambda s:(s['classification']!='featured',-s['n_events']))[:5]
        lines.append('主要信号：'+'；'.join(f"S{s['signal_id']} {s['f_lo_MHz']:.2f}–{s['f_hi_MHz']:.2f} MHz，BW {s['bandwidth_MHz']:.3f}，时长{s['duration_ms']:.3f} ms，T0 {s['T0_ms']:.3f} ms，{s['n_channels']}信道/间隔{s['spacing_MHz']:.3f} MHz，{s['classification']}({s['rule']})" for s in top)+'。')
    lines+=['## 检测复核与建议','随机三段/录制放大图已生成；人工复核结论见 verification.json。','背景里有规律的环境通信也可满足 featured；also_in_background 仅做形状和频率匹配，不代表无人机来源。','持续型 a 按出现窗口而非占空比判断，频繁噪点可能被提升；建议后续增加最小时长/面积或功率置信度，并用人工标注验证。','DBSCAN 会把宽带信号内部的孔洞和边缘碎片分成多类；试点不宜据此直接自动截频。','周期候选可能有谐波/次谐波歧义，需结合 IAT 图与信道访问顺序；本试点不据此证明协议类型。','原始文件计数/最新修改时间及 PNG 完整性核对见 verification.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':render()

