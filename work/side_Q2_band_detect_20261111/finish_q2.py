"""Statistics, figures and integrity verification; run only after calibration lock."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE';os.environ['PYTHONIOENCODING']='utf-8'
import importlib.util,sys,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('q2',OUT/'run_q2.py');q=importlib.util.module_from_spec(sp);sp.loader.exec_module(q)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei'];plt.rcParams['axes.unicode_minus']=False
def stat_tables():
    summaries=[]
    for k in q.KS:
        for g in q.GS:
            t=q.tag(k,g);seg=pd.read_csv(OUT/f'segments_{t}.csv');bands=pd.read_csv(OUT/f'bands_{t}.csv')
            note='k*未用5.8GHz真实噪声校准；跨接收机适用性是假设，非验证'
            seg['threshold_calibration_note']=np.where(seg.band=='5.8GHz',note,'2.4GHz真实接收机噪声及合成噪声校准；四候选均未全部达标，按规则回退6dB')
            bands['threshold_calibration_note']=bands.clip_id.map(seg.set_index('clip_id').threshold_calibration_note)
            seg.to_csv(OUT/f'segments_{t}.csv',index=False,encoding='utf-8-sig')
            bands.to_csv(OUT/f'bands_{t}.csv',index=False,encoding='utf-8-sig')
            assert len(seg)==1426 and seg.clip_id.is_unique
            rows=[]
            def add(metric,values,**group):
                v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
                rows.append(dict(metric=metric,**group,n=len(v),**{f'q{int(z*100):02d}':float(np.quantile(v,z)) if len(v) else np.nan for z in (0,.5,.9,.95,1)}))
            for keys,v in seg[seg.label=='drone'].groupby(['source','band','model','state','bw'],dropna=False):
                group=dict(zip(['source','band','model','state','bw'],keys));add('coverage',v.coverage,**group);add('band_fraction',v.band_fraction,**group)
            for keys,v in seg.groupby(['source','band','label']):
                group=dict(zip(['source','band','label'],keys))
                for m in ('band_count','box_count','band_fraction'): add(m,v[m],**group)
            merged=bands.merge(seg[['clip_id','source','band','label']],on='clip_id',validate='many_to_one')
            for keys,v in merged.groupby(['source','band','label']):
                group=dict(zip(['source','band','label'],keys))
                add('boxes_per_band',v.box_count,**group);add('band_width_mhz',v.width_mhz,**group)
                rows.append(dict(metric='width_lt_1_fraction',**group,n=len(v),value=float((v.width_mhz<1).mean())))
            bg=merged[(merged.source=='raw')&(merged.label=='ambient')].copy()
            overlap=np.zeros(len(bg),bool);overwidth=np.zeros(len(bg))
            # WiFi channel intervals are disjoint, so pairwise intersections sum directly.
            for fc in (2412,2437,2462):
                w=np.maximum(0,np.minimum(bg.f_end_mhz.to_numpy(),fc+11)-np.maximum(bg.f_start_mhz.to_numpy(),fc-11))
                overlap|=w>0;overwidth+=w
            bg['wifi_overlap']=overlap;bg['wifi_overlap_width_mhz']=overwidth
            for band,v in bg.groupby('band'):
                rows.append(dict(metric='background_wifi_band_fraction',band=band,n=len(v),value=float(v.wifi_overlap.mean())))
                rows.append(dict(metric='background_wifi_width_fraction',band=band,n=len(v),value=float(v.wifi_overlap_width_mhz.sum()/v.width_mhz.sum()) if v.width_mhz.sum() else np.nan))
            background=seg[(seg.source=='raw')&(seg.label=='ambient')].copy()
            counts=bg.groupby('clip_id').wifi_overlap.agg(['sum','count'])
            background=background.merge(counts,left_on='clip_id',right_index=True,how='left')
            background['wifi_band_overlap_fraction']=background['sum']/background['count']
            background.to_csv(OUT/f'stats_background_segments_{t}.csv',index=False,encoding='utf-8-sig')
            bg.to_csv(OUT/f'stats_background_bands_{t}.csv',index=False,encoding='utf-8-sig')
            stats=pd.DataFrame(rows)
            stats['threshold_calibration_note']=np.where(stats.band=='5.8GHz',note,'')
            stats.to_csv(OUT/f'stats_{t}.csv',index=False,encoding='utf-8-sig')
            s=dict(k=k,g=g,boxes=int(seg.box_count.sum()),bands=len(bands),median_bands=float(seg.band_count.median()),p90_bands=float(seg.band_count.quantile(.9)),
              median_width=float(bands.width_mhz.median()),narrow_fraction=float((bands.width_mhz<1).mean()),zero_reference=int(((seg.label=='drone')&(seg.reference_total==0)).sum()))
            for src in ('data','raw'):
                v=seg[(seg.source==src)&(seg.label=='drone')]
                s[src+'_coverage_median']=float(v.coverage.median());s[src+'_coverage_min']=float(v.coverage.min());s[src+'_band_fraction_median']=float(v.band_fraction.median())
            summaries.append(s)
    summary=pd.DataFrame(summaries);summary['threshold_calibration_note']='5.8GHz：k*未用真实噪声校准；跨接收机适用性是假设，非验证'
    summary.to_csv(OUT/'stats_summary_all_k.csv',index=False,encoding='utf-8-sig')
    ks=q.load('threshold_lock.json')['k_star'];boxes=pd.read_csv(OUT/f'boxes_k{ks}.csv')
    boxes['threshold_calibration_note']=boxes.clip_id.map(seg.set_index('clip_id').threshold_calibration_note)
    boxes.to_csv(OUT/f'boxes_k{ks}.csv',index=False,encoding='utf-8-sig')
    return summaries
def render_rows(items,name,k,g,plotindex,category):
    fig,axes=plt.subplots(len(items),2,figsize=(19,3.2*len(items)),squeeze=False)
    for r,(c,d,boxes,bands) in enumerate(items):
        f=d['f'];df=f[1]-f[0];extent=[f[0]-df/2,f[-1]+df/2,0,len(d['S'])*d['dt']*1000]
        for col,ax in enumerate(axes[r]):
            im=ax.imshow(d['raw']-d['F'],origin='lower',aspect='auto',extent=extent,vmin=-5,vmax=35,cmap='viridis',rasterized=True)
            ax.set_xlabel('频率（MHz）');ax.set_ylabel('片段时间（ms）')
            title=f"{c['clip_id']} {c.get('scene',c.get('cal_group',''))} {c.get('model','噪声校准')} {c.get('band','')}"
            ax.set_title(title+('｜原始时频图' if col==0 else f'｜k*={k} dB，g={g:g} MHz，{len(bands)}带 / {len(boxes)}框'),fontsize=9,pad=28)
            if col:
                for b in bands:
                    ax.add_patch(Rectangle((b['f_start_mhz'],0),b['width_mhz'],extent[3],fill=False,edgecolor='#ff9a00',linewidth=2*72/85))
                    ax.text((b['f_start_mhz']+b['f_end_mhz'])/2,1.01,f"#{b['band_id']} {b['f_start_mhz']:.2f}–{b['f_end_mhz']:.2f} MHz",transform=ax.get_xaxis_transform(),fontsize=5.5,rotation=12,ha='center',va='bottom',clip_on=False)
                if 0<len(boxes)<=200:
                    vertices=[]
                    for b in boxes:
                        x0=b['f_start_mhz'];x1=b['f_end_mhz'];y0=b['t_start_ms'];y1=b['t_end_ms']
                        vertices.append([(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)])
                    ax.add_collection(LineCollection(vertices,colors='red',linewidths=.4))
        plotindex.append(dict(figure=name,row=r+1,clip_id=c['clip_id'],category=category,k=k,g=g,band_count=len(bands),box_count=len(boxes)))
    fig.subplots_adjust(left=.055,right=.94,bottom=.045,top=.95,hspace=.65,wspace=.15)
    cb=fig.colorbar(im,cax=fig.add_axes([.955,.15,.008,.7]));cb.set_label('相对逐频率底噪（dB），统一 −5～35')
    fig.savefig(OUT/name,dpi=85);plt.close(fig)
    assert (OUT/name).stat().st_size<=5_000_000
    q.resources()
def figures():
    ks=q.load('threshold_lock.json')['k_star'];clips=q.old('clips.json');byid={c['clip_id']:c for c in clips};index=[]
    for g in q.GS:
        t=q.tag(ks,g);seg=pd.read_csv(OUT/f'segments_{t}.csv');selected=q.old('selected_examples.json')
        selected['最低覆盖率']=seg[(seg.label=='drone')&seg.coverage.notna()].sort_values(['coverage','clip_id']).head(4).clip_id.tolist()
        selected['最高频带占比']=seg.sort_values(['band_fraction','clip_id'],ascending=[False,True]).head(4).clip_id.tolist()
        for category,ids in selected.items():
            for offset in range(0,len(ids),6):
                items=[]
                for cid in ids[offset:offset+6]:
                    c=byid[cid];c=dict(c,scene=q.io.scene_for(c));d=q.grid(c);boxes=q.detect(d,ks);bands=q.io.merge(boxes,g,len(d['S']));items.append((c,d,boxes,bands))
                name=f'图_{t}_{category}_{offset//6+1:02d}.png';render_rows(items,name,ks,g,index,category)
                print('figure',name,flush=True)
    # Generate exact calibration examples again using the identical RNG stream.
    wanted={};rng=np.random.default_rng(20261111)
    for c in q.load('calibration_clips.json'):
        x=None
        if c['cal_group']=='c': x=(rng.standard_normal(c['length'],dtype=np.float32)+1j*rng.standard_normal(c['length'],dtype=np.float32))/np.sqrt(2)
        key=c['cal_group'];items=wanted.setdefault(key,[])
        # c shows one segment for each synthetic grid, a uses distinct recordings, b distinct files.
        signature=(c['fs'] if key=='c' else c['parts'][0]['path'])
        if len(items)>=2 or any(z[4]==signature for z in items): continue
        d=q.grid(c,c['cal_bounds'],c['cal_n'],x);boxes=q.detect(d,ks);bands=q.io.merge(boxes,.5,len(d['S']));items.append((c,d,boxes,bands,signature))
    items=[v[:4] for group in ('a','b','c') for v in wanted[group]]
    render_rows(items,'图_纯噪声校准.png',ks,.5,index,'纯噪声校准')
    pd.DataFrame(index).to_csv(OUT/'plot_index.csv',index=False,encoding='utf-8-sig')
    return len(set(x['figure'] for x in index)),len(index)
def verify_and_report(summaries,nfig,nrows):
    before=q.load('protected_before.json');after=q.io.snapshot();q.dump('protected_after.json',after);assert before==after
    ks=q.load('threshold_lock.json')['k_star'];met=q.load('detection_complete.json')
    verify=dict(protected_identical=before==after,protected_files=len(after),peak_bytes=max(q.peak,met['peak_bytes'],q.load('threshold_lock.json')['peak_bytes']),
       figures=nfig,plot_rows=nrows,cache_bytes=0,report_lines=0)
    assert verify['peak_bytes']<8*1024**3
    # Keep the original calibration record at the beginning without changing any selected threshold.
    report=(OUT/'REPORT.md').read_text(encoding='utf-8').splitlines()
    report+=['','主表使用预先锁定的k*；其他k仅见stats_summary_all_k.csv与各附表。',
      '|g MHz|框总数|带总数|每段带数中位/P90|实验室覆盖中位/最小|实验室带占比中位|城区覆盖中位/最小|城区带占比中位|窄带占比|零参照段|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['k']!=ks:continue
        report.append(f"|{s['g']:g}|{s['boxes']}|{s['bands']}|{s['median_bands']:g}/{s['p90_bands']:g}|{s['data_coverage_median']:.4%}/{s['data_coverage_min']:.4%}|{s['data_band_fraction_median']:.4%}|{s['raw_coverage_median']:.4%}/{s['raw_coverage_min']:.4%}|{s['raw_band_fraction_median']:.4%}|{s['narrow_fraction']:.4%}|{s['zero_reference']}|")
    report+=['', '处理1426段：实验室576段，城区850段；片段、排除清单、例图选择和背景参照直接复用Q目录对应JSON/CSV，未重抽样。',
      '检测在线性功率上时间16×频率4均值平滑（nearest），转dB后逐频率时间P20；全局取有效频点中位，min(n,G+3)后频率17点中值（nearest）。',
      '固定谱线±0.3MHz：平滑前线性功率以有效像素P20替换防止扩散；不参与全局底噪，n置G后滤波，S置F；形态学后再清零。',
      '闭运算时间9×频率3，开运算时间3×频率2；四邻域，保留面积≥50；框均值和峰值为连通域像素S−F。',
      '按框频率区间间隙严格小于g合并；带占空比为框时间区间并集/全时长；频带宽度含合并间隙，区间半开。',
      '实验室覆盖权重为S>F+10像素的max(10^(S/10)−10^(F/10),0)，对时间求和；覆盖为频带内权重/全部权重。',
      '城区覆盖按Q的1ms最大功率占用率减同中心背景P90、负值置零；参照为0时coverage空值，不进入覆盖率统计或最低覆盖图。',
      '主分析频域、Hann窗及无重叠1024/2048网格照搬Q；校准b为125MS/s、1024；合成两网格为122.88MS/s、1024及250MS/s、2048。',
      '校准c两网格各60段20ms，合成IQ再做Hann FFT；a按2343–2393MHz，b按2345–2395MHz；宽度分母使用指定分析端点差。',
      'stats_k*_g*.csv含来源/频段/机型/状态/带宽的覆盖q00/q50与带占比q50，来源/频段/标签的带数和框数q50/q90/q100。',
      'stats_background_segments记录每段背景带数、框数、带占比、WiFi重叠比例；stats_background_bands记录逐带宽度与WiFi交叠宽度。',
      'WiFi按2412/2437/2462±11MHz任一区间相交；统计同时给出相交频带数比例与交叠宽度比例，按2.4/5.8GHz分别列出。',
      '碎片化统计列每带框数分布及宽度<1MHz比例；boxes只保存k*一份，两档g共享相同框；其余k只保存频带和片段汇总。',
      '检测四个k用四线程独立计算；8组参数各80段与串行汇总逐项一致；serial_partial_preserved保留重跑前部分输出，未删除文件。',
      f'图{nfig}张、例图行{nrows}；每张≤6行、85dpi、≤5MB；左右用同一未平滑时频矩阵及F，统一−5至35dB；空心频带框2像素，红色框仅在≤200时绘制。',
      f'保护目录{len(after)}个文件，前后SHA256相同={before==after}；详见protected_before/after.json。',
      f'进程峰值内存{verify["peak_bytes"]/1024**3:.3f}GiB；缓存0字节；校准{q.load("threshold_lock.json")["calibration_seconds"]:.1f}s；检测{met["seconds"]:.1f}s。',
      '运行使用D:/anaconda/python.exe -I；KMP_DUPLICATE_LIB_OK=TRUE、PYTHONIOENCODING=utf-8。仅以只读方式打开G:/drone/data与raw；未读取F盘、未训练、未装包、未删除、未提交。']
    assert len(report)<=80;verify['report_lines']=len(report)
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8');q.dump('verification.json',verify)
    print(json.dumps(verify),flush=True)
if __name__=='__main__':
    summaries=stat_tables();nfig,nrows=figures();verify_and_report(summaries,nfig,nrows)
