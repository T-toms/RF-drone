from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pilot import *

def finish():
    results=json.loads((OUT/'results.json').read_text(encoding='utf8'))
    v=json.loads((OUT/'verification.json').read_text(encoding='utf8'))
    spans=pd.read_csv(OUT/'spans.csv'); ss=[s for r in results for s in r['signals']]
    hop=[s for s in results[1]['signals'] if s['classification']=='featured' and ('c' in s['rule'] or ('b' in s['rule'] and s['n_channels']>=3))]
    candidates=[s for s in hop if 6<=s['n_channels']<=8 and abs(s['T0_ms']-2.5)<=.15 and all(any(abs(c['center']-f)<=2 for c in s['channel_metrics'].values()) for f in [5747,5762,5772])]
    video=[s for s in results[0]['signals'] if s['classification']=='featured' and any(x in s['rule'] for x in ['a','b']) and any(abs(c['center']-5786)<=5 and 15<=c['bw']<=28 for c in s['channel_metrics'].values()) and abs(s['T0_ms']-2.5)<=.15]
    checks=dict(recording_count_8=len(results)==8,target_002_single_hopping_pass=len(candidates)==1,target_002_candidate_ids=[s['signal_id'] for s in candidates],target_007_video_pass=bool(video),no_featured_channels_above_50=all(s['n_channels']<=50 for s in ss if s['classification']=='featured'),no_featured_duration_below_005ms=all(s['duration_ms']>=.05 for s in ss if s['classification']=='featured'))
    required=[OUT/f'{prefix}_{r["recording"]}.png' for prefix in ['annot','zoom','clusters'] for r in results]
    v['png_sizes']={p.name:p.stat().st_size for p in OUT.glob('*.png')}
    v['required_png_count']=len(required)
    v['pngs_complete']=len(required)==24 and all(p.exists() and p.stat().st_size>30000 for p in required)
    v.update(checks);v['source_after']=snapshot();v['source_unchanged']=v['source_before']==v['source_after'];dump(OUT/'source_after.json',v['source_after'])
    notes_path=OUT/'visual_review.json'
    if notes_path.exists():
        notes=json.loads(notes_path.read_text(encoding='utf8'))
        for entry in v['random_views']:
            note=notes.get(entry['recording'])
            if note:
                entry['visual_check']='reviewed_macro_structure'
                entry['visual_note']=note['note']
                manual=note.get('manual_counts',{}).get(str(entry['chunk']))
                entry['manual_count']=manual
                if manual is not None:
                    difference=abs(entry['n_intersecting_events']-manual)/max(manual,1)
                    entry['relative_difference']=difference
                    entry['count_agreement_pass']=difference<=.1
    # Never substitute algorithm counts for an independent visual census.
    v['manual_count_acceptance_pass']=all(e.get('count_agreement_pass',False) for e in v['random_views'])
    v['manual_review_limitation']='逐窗人工数为null表示未能完成独立逐个计数；不以程序数代替人工数。'
    v['acceptance_status']='passed' if all(checks[k] for k in checks if k.endswith('pass') or k.startswith('no_')) and v['manual_count_acceptance_pass'] and v['source_unchanged'] and v['pngs_complete'] else 'failed_acceptance'
    v['method_notes']=['8连通从10dB种子生长到5dB；物理闭运算0.05ms×1MHz；闭运算后挖稳定细线±0.3MHz/中心±0.5MHz。','归并只将非大突发并入原始大突发；多个候选选择频率距离最近者，平局按原始检测顺序；不跨v3块。','energy以超floor的线性相对功率像素求和；area为分量有效像素数；a按时间相交比例分摊面积，属于窗内面积近似。','逐文件STFT不补跨文件滑窗，边界按3hop和频率接触合并，parts记录；继承G限制。','DBSCAN仅标准化log带宽和log时长，eps=.07/min_samples=5，精确邻域；周期候选继承G量化自相关法。','序列在同一块内比较，不跨NN；最短匹配率>=.9周期，否则报告最大匹配候选，并非已确认周期。']
    v['cross_boundary_confirmation']=json.loads((OUT/'boundary_evidence.json').read_text(encoding='utf8')) if (OUT/'boundary_evidence.json').exists() else None
    if v['cross_boundary_confirmation']:
        v['cross_boundary_confirmation']['visual_pass']=True
        v['cross_boundary_confirmation']['visual_note']='???boundary_merge.png?50ms???????parts=2???????????/?????????????????????????????????'
    dump(OUT/'verification.json',v)
    old=pd.read_csv(ROOT/'work/burst_regularity_pilot_20261106/spans.csv')
    lines=['# G2 规律性试点修订（2026-11-06）','仅指定8个录制；原始数据只读；未截频、切片、训练、划分、装包、调用DSH或提交。','检测：1024 Hann/hop512；种子+10dB、生长+5dB、8连通；闭运算0.05ms×1MHz；稳定细线±0.3MHz、中心±0.5MHz在闭运算后挖除。','归并/碎片：大突发≥1MHz且≥0.1ms；邻近小块±0.05ms/2MHz归并；之后<0.05ms或<0.3MHz或能量<P5为fragment，不聚类分类。','聚类：标准化log(BW)/log(duration)，DBSCAN eps=.07,min_samples=5；功率只参考。a按窗内有效像素面积≥中位单次面积0.5倍，50ms率≥.9。','b/c沿G；5–19事件且IAT CV≤.2优先记sparse_periodic/review；序列最短匹配≥.9，否则仅最大匹配候选。所有指标见signals.csv。','清单下列每项：S编号 频率范围MHz / BW MHz / 时长ms / T0 ms / 信道中心MHz / 序列(跳,ms,匹配) / 分类。']
    for r in results:
        sp=spans[spans.recording==r['recording']].iloc[0];op=old[old.recording==r['recording']].iloc[0]
        lines.append(f"## #{r['index']} {r['recording']}")
        listing=[]
        for s in r['signals']:
            if s['n_events']<5:continue
            centers=','.join(f"{c['center']:.2f}" for c in s['channel_metrics'].values())
            listing.append(f"S{s['signal_id']} {s['f_lo_MHz']:.2f}–{s['f_hi_MHz']:.2f}/{s['bandwidth_MHz']:.2f}/{s['duration_ms']:.3f}/{s['T0_ms']:.3f}/[{centers}]/({s['hop_seq_period_hops']},{s['hop_seq_period_ms']},{s['hop_seq_match']})/{s['classification']}({s['rule']})")
        lines.append('信号：'+'；'.join(listing) if listing else '信号：无非fragment突发。')
        lines.append(f"fragment={r['fragment_count']}；featured span/total_bw/plus_review_span={sp.featured_span_MHz:.3f}/{sp.featured_total_bw_MHz:.3f}/{sp.featured_plus_review_span_MHz:.3f} MHz；G span/total_bw={op.featured_span_MHz:.3f}/{op.featured_total_bw_MHz:.3f} MHz。")
        vv=[e for e in v['random_views'] if e['recording']==r['recording']]
        lines.append('??????chunk:??/??/?????'+'?'.join(f"{e['chunk']}:{e['n_intersecting_events']}/{e.get('manual_count')}/{e.get('relative_difference')}" for e in vv)+'?'+(vv[0].get('visual_note','???????') if vv else '????'))
    sparse=[f"#{r['index']}/S{s['signal_id']} n={s['n_events']}, IAT_CV={s['iat_cv']:.3f}" for r in results for s in r['signals'] if s['sparse_periodic']]
    lines+=['sparse_periodic：'+('；'.join(sparse) or '无满足条件者；这不证明不存在稀疏规律信号。'),'验收：'+json.dumps(checks,ensure_ascii=False),'人工计数：'+str(v['manual_count_acceptance_pass'])+'；逐窗程序数、人工数及差异见verification.json，null不算通过。','原始快照一致='+str(v['source_unchanged'])+'；24张必需PNG齐全且>30KB='+str(v['pngs_complete'])+'。','方法局限：逐文件STFT未补跨边界滑窗；面积按事件内部均匀时间分布分摊；中心掩膜可能把横跨中心的宽带突发分成两半，外框频带含屏蔽孔。','阈值仍需结合物理突发核对调整；不为满足预期信道数而手动合并不同形状簇，不能将候选周期视为协议确认。','最终状态：'+v['acceptance_status']+'；未通过项保留原标准，不能作为自动截频已验收依据。']
    assert len(lines)<=60
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status=v['acceptance_status'],checks=checks,report_lines=len(lines),signals=len(ss)),ensure_ascii=False),flush=True)
if __name__=='__main__':finish()
