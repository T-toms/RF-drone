"""Finish Task E from saved measurements, without remeasuring checkpoints."""
import sys,json,csv,re,hashlib,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_analysis import OUT,csvout,dump,snapshot

def read(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def main():
    cps=[json.loads(p.read_text(encoding='utf-8')) for p in OUT.glob('checkpoint_*.json')]
    baseline=json.loads((OUT/'resume_baseline.json').read_text());unchanged=all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==v for n,v in baseline.items())
    vals=read('cancel_validation.csv'); lines=read('lines_by_recording.csv'); classified=[]; home=[]
    for s in cps:
        r=s['record'];ls=s['lines'];
        for l in ls:
            stable=l['presence_rate']>=.9
            v=[v for v in vals if v['recording']==r['key'] and abs(float(v['freq_mhz'])-l['freq_mhz'])<1e-6 and v['method'] in ('cancel','cancel_short')]
            # Persistence alone does not establish oscillator origin. Failures remain explicit.
            methods=collections.defaultdict(list)
            for a in v:methods[a['method']].append(a)
            passed=any(len(a)>=3 and all(x['passed'] in ('True',True) for x in a) for a in methods.values())
            nv=[v for v in vals if v['recording']==r['key'] and abs(float(v['freq_mhz'])-l['freq_mhz'])<1e-6 and v['method']=='notch']
            notch_ok=len(nv)>=3 and all(x['passed']=='True' for x in nv)
            classified.append(dict(**l,notch_validated=notch_ok,classification='persistent_tone_candidate' if stable else 'transient_peak_unconfirmed',automatic_removal_allowed=stable and passed,origin='unidentified',cancellation_validated=passed))
        if r['label']=='background' and r['key'].startswith('20260710-11'):
            home.append(dict(recording=r['key'],candidates=len(ls),persistent=sum(l['presence_rate']>=.9 for l in ls),median_presence=float(np.median([l['presence_rate'] for l in ls])) if ls else 0,median_width_khz=float(np.median([l['width_khz'] for l in ls])) if ls else 0))
    csvout('lines_classified.csv',classified);csvout('home_line_audit.csv',home)
    import matplotlib.pyplot as plt
    chosen=[r for r in home if r['candidates']>=100]
    fig,axes=plt.subplots(len(chosen),1,figsize=(13,3*len(chosen)),squeeze=False)
    for ax,r in zip(axes[:,0],chosen):
        spec=np.load(OUT/('spectrum_'+r['recording']+'.npz'));f=spec['freq'];db=10*np.log10(spec['power']+1e-30)
        ax.plot(f,db,lw=.6,color='steelblue');a=[l for l in classified if l['recording']==r['recording']]
        for stable,color,label in [(False,'orange','Transient candidates'),(True,'crimson','Persistent candidates')]:
            aa=[l for l in a if (l['presence_rate']>=.9)==stable];ff=[l['freq_mhz'] for l in aa]
            ax.scatter(ff,np.interp(ff,f,db),s=14,color=color,label=label)
        ax.set_title(r['recording']+'; candidates='+str(r['candidates'])+'; persistent='+str(r['persistent']));ax.set_xlabel('Absolute frequency (MHz)');ax.set_ylabel('PSD (dB/Hz)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'home_line_audit.png',dpi=140);plt.close(fig)
    lookup={(l['recording'],round(l['freq_mhz'],6)):l for l in classified}
    plans=read('crop_plan.csv')
    for p in plans:
        actions=json.loads(p['lines'])
        for a in actions:
            l=lookup[(p['recording'],round(a['freq_mhz'],6))];a['classification']=l['classification']
            allowed=l['classification']=='persistent_tone_candidate' and (l['notch_validated'] if a['method']=='notch' else l['cancellation_validated'])
            if a['method']!='removed_by_bandpass' and not allowed:
                a['proposed_method']=a['method'];a['method']='review_required';a['reason']='transient_or_cancellation_not_validated'
        p['center_status']='review_required_candidate_mask' if p['label']=='drone' and any(a['classification']=='transient_peak_unconfirmed' for a in actions) else 'planned'
        p['lines']=json.dumps(actions)
    csvout('crop_plan.csv',plans)
    review_count=sum(a['method']=='review_required' for p in plans for a in json.loads(p['lines']))
    sources=read('source_groups.csv');duplicates=[s for s in sources if s['exclude_reason']=='duplicate']; sources=[s for s in sources if s['exclude_reason']!='duplicate']
    csvout('source_groups.csv',sources)
    if duplicates:csvout('source_duplicates_excluded.csv',duplicates)
    verification=json.loads((OUT/'verification.json').read_text());v=verification['source_rows'];v.update(actual=len(sources),pass_=len(sources)==v['expected'],duplicate_rows_separate=len(duplicates));verification['resume']=dict(pass_=unchanged and len(cps)==189,restored=len(baseline),completed=len(cps),new=len(cps)-len(baseline));verification['raw_unchanged']['after']=snapshot();verification['raw_unchanged']['pass_']=verification['raw_unchanged']['before']==verification['raw_unchanged']['after']; verification['review_gate']=dict(pass_=all(a['method'] in ('review_required','removed_by_bandpass') or lookup[(p['recording'],round(a['freq_mhz'],6))]['classification']=='persistent_tone_candidate' for p in plans for a in json.loads(p['lines'])),review_actions=review_count); verification['png']=dict(pass_=all(p.stat().st_size>30000 for p in OUT.glob('*.png')),count=len(list(OUT.glob('*.png')))); dump('verification.json',verification)
    recs=json.loads((OUT/'recordings.json').read_text()); bg=read('background_sessions.csv');inventory=read('lines_inventory.csv'); scan=json.loads((OUT/'scan_verification.json').read_text());short=json.loads((OUT/'short_block_summary.json').read_text()) if (OUT/'short_block_summary.json').exists() else {}
    high=[r for r in home if r['candidates']>=100]; persistent=sum(l['classification']=='persistent_tone_candidate' for l in classified)
    group_summary=[]
    for group in sorted(set(l['group'] for l in classified)):
        a=[l for l in classified if l['group']==group];group_summary.append(dict(group=group,candidates=len(a),persistent=sum(l['classification']=='persistent_tone_candidate' for l in a),overlaps=sum(l['overlaps_signal'] for l in a),frequencies_mhz=json.dumps([round(l['freq_mhz'],6) for l in a])))
    csvout('group_summary.csv',group_summary)
    report=['# 任务 E：截频带前谱线与同源核查',
    f'已完成 {len(cps)}/189 个录制/会话；恢复原有 {len(baseline)} 个断点，仅新增 {len(cps)-len(baseline)} 个，原断点 SHA256 全部不变={unchanged}。',
    '未训练、未生成样本、未保存切片、未划分；G:/drone 只读；未调用 DSH、未发送 heartbeat。',
    '## 谱线测量与结论',
    f'65536 点 Hann、50% 重叠；局部±1MHz中位数突出≥10dB、−10dB宽≤50kHz、组内±20kHz合并；共{len(inventory)}条合并候选。',
    '均匀抽5个有效文件，每会话≥200ms（v3两文件约100ms不足）；10ms整块统计出现率、漂移与幅度波动。',
    '各组完整频率/中心频点/出现率见 lines_inventory.csv；group_summary.csv 给出分组清单，lines_by_recording.csv 给出逐录制重叠。',
    f'逐录制候选中{persistent}/{len(classified)}条出现率≥0.9；稳定仅表示单音候选，无法由IQ确定硬件本振归因。',
    '## 2026-07-11_home 的100–185条细线专项核查',
    '日期前缀2026-07-11_home对应原始会话20260710-11xxxx，不能按文件名前缀日期误选断点。',
    '长时谱局部峰判据不足以证明本振泄露：局部中位数会被宽带包络凹口拉低；突发谱峰也可满足50kHz宽度。',
    *[f'{r["recording"]}：{r["candidates"]}候选，稳定{r["persistent"]}，10ms出现率中位数{r["median_presence"]:.4f}，宽度中位数{r["median_width_khz"]:.2f}kHz。' for r in high],
    'home_line_audit.png显示候选簇集于多个较宽谱瓣；代表会话30kHz高斯平滑后的谱瓣宽度见home_lobe_audit.csv，多数约0.7–1.2MHz，不能把谱瓣顶部细峰当作独立本振。',
    '这些会话的百余峰主体不符合稳定本振/连续单音，属于瞬态窄峰误归为泄露候选；与突发宽带/调制信号一致，不能仅凭PSD证明具体为WiFi。',
    '修正：保留原始候选测量不删数；lines_classified.csv增加出现率≥0.9稳定性门槛，自动消除还要求至少3个验证块全部达标；其余crop_plan标review_required。',
    '低出现率也可能是间歇单音，故保留unconfirmed而非断言所有峰均为WiFi；不应对背景百余候选批量陷波。',
    '## 消除方法及验证',
    'lab_preproc.py：notch为复下变频后6阶Butter高通、双向零相位，半宽max(2×线宽,60kHz)；cancel为10ms块FFT插值、频率优化、复幅度最小二乘；零频另测均值相减。',
    f'验证{len(vals)}项，达标{sum(v["passed"]=="True" for v in vals)}；完整失败列表在cancel_validation.csv（residual≤3dB且|off_line_delta|≤0.1dB）。',
    f'稳定但未全达标的线已试1ms块：{short}；最终选择见cancel_final_parameters.csv；仍失败项禁止自动对消。',
    '续跑异常：refine_validation.py对无效清单行的空start_sample转int失败；新增run_refine_resume.py兼容入口，原脚本保留不改，仅有效文件进入验证。',
    '每录制每线抽3个10ms块；重叠/不重叠类别按实际存在情况验证；首3块可能无瞬态线，低残留不等于成功消除瞬态信号。',
    '重叠判据以线外150–300kHz邻域占用推断底层信号，存在边界不确定性；背景overlaps_signal=False只代表没有无人机标注，不代表没有其他无线电信号。',
    '## 截取规划',
    '固定输出61.44MS/s，通带±28MHz，过渡到±30.72MHz；data为1/2、v3为384/625重采样；合成±20/35MHz测试见crop_synthetic_tests.json。',
    'crop_plan.csv给出每录制中心、占用比例、每候选处理决策；过渡带内不能假定已被带通完全消除。',
    '中心与占用比例沿用原候选掩膜规划；有瞬态候选的无人机中心标center_status=review_required_candidate_mask，F前需复核，未为此重算已完成测量。',
    '占用比例<0.9录制：'+', '.join(r['key'] for r in recs if r['label']=='drone' and r['occupancy_fraction'] is not None and r['occupancy_fraction']<.9),
    '背景候选数、可用范围、覆盖宽度见background_sessions.csv；中心列表与56MHz绝对窗见crop_plan.csv。',
    'no_valid_center：'+', '.join(r['session'] for r in bg if r['candidate_count']=='0'),
    '## 同源分组与验收',
    f'source_groups.csv共{len(sources)}行、{len(set(s["group_id"] for s in sources))}组；duplicate的{len(duplicates)}行单列source_duplicates_excluded.csv，兼顾逐文件排除记录与验收非重复行数。',
    'data同录制所有epoch/seg同组；v3同raw session及_s0001/_s0002同组，扫频按中心频率细分；raw不另入表。',
    '任务F必须以group_id为最小单位划分：同组所有片段（含重叠片段）必须进入同一个训练/验证/测试集合。',
    'raw根目录20260709-030546_ambient、20260709-040756_mavic2与urban内重复；clean_export为空（任务给定）。',
    f'扫频对应：4/5重采样，100MS/s坐标偏移{scan["offset_samples"]}点，相关{scan["correlation"]:.10f}，通过={scan["passed"]}。',
    'verification.json：'+json.dumps({k:v.get('pass_',v.get('passed')) for k,v in verification.items()},ensure_ascii=False),
    '所有测量/消除失败按数值保留，任务E完成不表示所有线已满足自动清除条件；任务F需遵守review_required门禁。']
    assert len(report)<=80
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('finalized',len(cps),'checkpoints; report',len(report),'lines',flush=True)
if __name__=='__main__':main()
