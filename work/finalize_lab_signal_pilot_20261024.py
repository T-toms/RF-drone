from pathlib import Path
import csv,json,hashlib
import numpy as np
from PIL import Image
from drone_audit_support import replace_markdown_section

W=Path(__file__).parent;O=W/'long_signal_pilot_20261024';A=W/'audit_drone_20261023'
rows=json.loads((O/'summary.json').read_text(encoding='utf8'));assert len(rows)==4
sources={r['path']:r for r in csv.DictReader((A/'data_preprocessing_manifest.csv').open(encoding='utf-8-sig'))}
checks={};numerical=[];allpaths=[]
for r in rows:
    key=r['folder'];tag=key+'__'+r['session'];z=np.load(O/(tag+'_statistics.npz'));counts=z['valid_frame_counts'];mask=counts>0
    expected=0
    for source in r['source']:
        s=sources[source['path']];allpaths.append(source['path'])
        if s['analysis_eligible']=='True':
            lo=(int(s['start_sample'])+4095)//4096;hi=int(s['end_sample_exclusive'])//4096;expected+=max(0,hi-lo)
    checks[key+':frame_coverage']=int(counts.sum())==expected==r['valid_fft_frames']
    holes=np.zeros(4000,dtype=bool)
    for ordinal in r['missing_ordinals']:holes[ordinal*100:(ordinal+1)*100]=True
    checks[key+':missing_positions']=np.all(counts[holes]==0) and np.all(np.isnan(z['mean_db'][holes])) and np.all(np.isnan(z['maximum_db'][holes]))
    linear=10**(z['mean_db'][mask].astype(np.float64)/10)
    derived=(linear*counts[mask,None]).sum(axis=0)/counts[mask].sum()
    direct=z['mean_psd'].reshape(1024,4).mean(axis=1)
    rel=float(np.max(np.abs(derived-direct)/np.maximum(direct,1e-20)))
    checks[key+':spectrum_consistency']=rel<1e-4
    checks[key+':finite_present_bins']=np.isfinite(z['mean_db'][mask]).all() and np.isfinite(z['envelope_db'][mask]).all()
    for name in ['overview','zoom']:
        with Image.open(O/r[name]) as im:im.verify()
    checks[key+':no_frequency_or_clock_assumption']=r['sampling_rate_assumed'] is None and r['center_frequency_assumed'] is None
    sr=sources[r['zoom_source']];lo,hi=r['zoom_source_interval'];ordinal=int(sr['epoch'])*20+int(sr['seg'])
    checks[key+':zoom_coordinates']=int(sr['start_sample'])<=lo<hi<=int(sr['end_sample_exclusive']) and r['zoom_recording_interval']==[ordinal*6144000+lo,ordinal*6144000+hi]
    e=z['envelope_db'];med=[float(np.nanmedian(e[k*2000:(k+1)*2000])) for k in range(2)]
    numerical.append(dict(folder=key,session=r['session'],epoch_power_medians_db=med,epoch1_minus_epoch0_db=med[1]-med[0],spectrum_reconstruction_relative_error=rel))
checks['source_files_unique']=len(allpaths)==len(set(allpaths))==157
checks['processed_samples']=sum(r['fft_samples_processed'] for r in rows)==964608000
checks={name:bool(value) for name,value in checks.items()}
assert all(checks.values()),checks
verification=dict(passed=True,checks=checks,files=157,samples_processed=964608000,figures_visually_reviewed=8,numerical=numerical)
(O/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
hashes={name:hashlib.sha256((A/name).read_bytes()).hexdigest() for name in ['data_preprocessing_manifest.csv','data_usable_runs.json','user_confirmations.json']}
(O/'source_evidence_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
notes={
 '009_2.4GHZ_10M-signal':'宽窄不同的突发分布于多个频率位置，局部有同时出现的成分；epoch切换前后功率分布有变化。',
 '009_2.4GHZ_60M-signal':'较宽频率范围的能量块与较窄突发交替出现，局部有明显低能量间隔；不能据此认定无人机缺失。',
 '007_5.8GHz_10M-signal':'相对频率约−0.16处有反复出现的带状突发，其他频率位置还有稀疏突发；局部可见特殊斜线结构，暂不据此贴协议标签。',
 'F009_2.4GHZ_60M-signal':'宽带能量块和较窄突发交替出现；3个已知缺段位置清楚留空。'
}
lines=[]
for r,n in zip(rows,numerical):
    ov=(O/r['overview']).resolve().as_posix();zo=(O/r['zoom']).resolve().as_posix()
    lines.append(f"|{r['folder']}|{r['files']}|{n['epoch1_minus_epoch0_db']:+.2f}|[概览]({ov}) / [局部]({zo})|")
section='''## 12. 首轮长时间信号观察

完成日期：2026-10-24。按用户要求先观察信号，精确中心频率问题暂不作为阻塞项。本轮覆盖4次代表录制、157个不同源段、964,608,000个复数样点，已生成4张概览和4张局部放大图；只覆盖mini5pro与avata2的部分配置，尚未代表全部13个机型。

### 12.1 看图方式与处理范围

每张概览依次显示：整次录制的分帧平均功率谱、长时平均谱、长时峰值谱、功率包络。概览采用4096点Hann窗、4096点步长，所有可用完整帧均参与；15帧聚合为一个显示时间格，即61,440个采样点。频率显示聚合至1024格。峰值谱有助于看见稀疏能量，但图像显示经过聚合，不能据其精确边缘测最短突发时长。

横轴为原始录制的百万采样点，纵轴为归一化频率cycles/sample；0.1表示采样率的0.1倍频率偏移。未假定fs或精确接收中心，没有换算成秒或绝对MHz。白色虚线为epoch0/1编号边界，灰色为缺段或被排除的无效数值，深色仅表示相对能量低。各录制颜色参照各自统计值，不能据不同图的颜色比较绝对接收功率。

局部图从谱变化较大的区域选取262,144点，采用2048点Hann窗、512点步长重新分析。所用源文件及原始样点区间记录在每组summary.json中，未从PNG反推IQ。

### 12.2 图像与初步观察

下表功率差是同录制epoch1与epoch0各时间格平均数字功率的中位数之差，不是校准接收功率或SNR。

|录制配置|源段数|epoch1−epoch0数字功率中位数差/dB|图像|
|---|---:|---:|---|
'''+ '\n'.join(lines)+'\n\n'+ '\n\n'.join('**'+r['folder']+'**：'+notes[r['folder']] for r in rows)+'''

F009的缺段为epoch0/seg04、epoch0/seg15、epoch1/seg05，均在原始样点坐标处留空，没有压缩时间轴或拼接跨越。两组009待机录制的epoch功率中位数分别相差约2.81、3.90 dB；仅记录变化，原因尚未确定，不能直接归因于增益调整或无人机状态改变。

这些图已经支持先研究候选频带、不同宽度突发、稀疏成分和局部低活动间隔。现阶段未把任何亮区自动确认为图传/控制链路，也未把暗区标为无人机不在场；尚需跨录制和背景对照。未设定筛窗阈值、未删除低能量片段、未训练模型。

### 12.3 核对与下一步

源段覆盖、缺段坐标、频谱聚合一致性、局部坐标和未使用采样率假设的检查均通过；8张图已目视检查。证据位于work/long_signal_pilot_20261024/：summary.json、每组statistics.npz、verification.json与source_evidence_hashes.json。初次运行缺段处出现All-NaN聚合提示，输出缺段标记正确；脚本已改为只在有数据的时间格上选择局部区域。

下一步适合把同样的观察扩展到其余机型/配置，再加入raw配对背景对照。先积累可追溯的现象记录，再决定哪些频带与时间区间值得进一步分离或标注。
'''
p=W.parent/'reports/实验室数据复核与改进方案_20261023.md';text=p.read_text(encoding='utf8')
text=text.replace('以下是接下来的工作安排，尚未完成长信号图谱或开展训练。','以下为调研工作安排；首轮4组长信号观察已完成（见第12节），尚未覆盖全部机型，尚未开展训练。')
p.write_text(replace_markdown_section(text,'## 12. 首轮长时间信号观察',section),encoding='utf8')
print(json.dumps(dict(passed=True,checks=len(checks),source_files=157,samples=964608000,epoch_differences={r['folder']:round(r['epoch1_minus_epoch0_db'],3) for r in numerical}),ensure_ascii=False,indent=2))
