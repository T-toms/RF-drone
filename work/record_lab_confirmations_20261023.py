from pathlib import Path
import json,csv,collections
W=Path(__file__).parent;O=W/'audit_drone_20261023';R=W.parent/'reports/实验室数据复核与改进方案_20261023.md'
facts={'source':'用户于2026-10-23转述导师确认','scope':'G:/drone/data，同一录制组内的分段，不跨不同录制；不套用到raw的独立采集文件',
 'filename_authoritative':True,'20M_means':'signal bandwidth 20 MHz','continuous_can_concatenate':True,
 'combination':'unknown','sample_rate_hz':None,'exact_receiver_center_hz':None,
 'join_policy':'同一目录标签和录制时间戳内按epoch、seg顺序逻辑拼接；缺段保留断点；副本不重复接入；无效零前缀保留位置标记'}
if (O/'user_confirmations.json').exists():
 old_facts=json.loads((O/'user_confirmations.json').read_text(encoding='utf8'));facts={**old_facts,**facts}
(O/'user_confirmations.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf8')
s=R.read_text(encoding='utf8')
s=s.replace('|seg与epoch连续性|文件名只给编号，没有逐段硬件时间戳|切段脚本、采样丢包/断点记录|','|seg与epoch连续性|用户转述导师确认信号连续、可以拼接；同录制按epoch/seg排序|已确认，依据为导师说明；四处缺段仍保留断点|')
s=s.replace('data带宽标签“图传设置”也仍应保留为待采集者确认的解释。','用户转述导师确认以文件名为准，20M就是信号带宽20 MHz；该字段不表示采样率，亦不表示已逐文件测得恰好20 MHz占用带宽。combination仍不确定。')
s=s.replace('向采集者仅补问data的fs/fc/有效带宽、combination、分段连续性与全零原因。','data的分段连续性与20M含义已获导师确认；剩余需补data的fs/精确接收fc/有效带宽、combination与全零原因。')
s=s.replace('下图横轴只是按seg顺序累计的零样点数量，不表示已证明片段之间物理连续。','下图横轴按seg顺序累计零样点数量；同录制的可拼接连续性现由用户转述导师确认，时间长度仍因未知fs而不换算。')
note='''
### 9.0 导师补充确认与逻辑拼接索引

用户于2026-10-23转述导师：data以文件名为准；信号连续、可以拼起来；20M就是信号带宽。本文据此将20M记录为20 MHz信号带宽，并将同一录制组内分段连续性从待确认改为导师确认。来源是采集方说明，不冒称是从MAT内部测得的参数。combination仍不确定；文件名没有明确采样率，不能把20 MHz带宽等同于20 MS/s采样率。

已生成data_continuous_runs.json/csv：按“目录标签＋录制时间戳”分组、按epoch/seg排序，副本不接入，缺段处另起连续区间；没有物理重写大体积IQ。该索引用于恢复观测顺序，不是训练划分；用户已明确原train/test仅为存放目录；它们都进入同一源数据池，后续由研究者重新划分。零前缀另带坐标，不直接贴成无线静默标签。

'''
if '### 9.0 导师补充确认' not in s:s=s.replace('### 9.1 背景缓存已解码',note+'### 9.1 背景缓存已解码')
R.write_text(s,encoding='utf8')
p=W/'finalize_continued_drone_audit_20261023.py';t=p.read_text(encoding='utf8')
t=t.replace('data的采样率、精确接收中心频率/有效带宽、combination语义、seg/epoch是否连续、零前缀生成原因，以及缺失机型是否补采，仍需要本批次采集配置或采集者说明。20M最合理的解释仍是20 MHz的带宽档位标签；目前不足以确认它是实际占用带宽、图传配置带宽还是其他命名约定，更不能据此当成20 MS/s采样率。','导师已确认data以文件名为准、信号连续可拼接、20M表示20 MHz信号带宽（见9.0节）。仍未知的是data采样率、精确接收中心频率/有效带宽、combination语义、零前缀原因及缺失机型是否补采；需要对应采集配置或采集者说明。带宽不是采样率，不能用20M换算10/20 ms样点数。')
p.write_text(t,encoding='utf8')
p=W/'deepseek-context.md';t=p.read_text(encoding='utf8');note='\n\n2026-10-23用户转述导师补充确认：data以文件名为准，20M是信号带宽20 MHz，信号连续可拼接；combination仍未知。同录制按epoch/seg排序拼接，缺段保留断点，不能据带宽推断fs。证据与逻辑索引见audit_drone_20261023/user_confirmations.json和data_continuous_runs.json。\n'
if '2026-10-23用户转述导师补充确认' not in t:p.write_text(t+note,encoding='utf8')
rows=[r for r in csv.DictReader((O/'data_index.csv').open(encoding='utf-8-sig')) if r['duplicate']=='False']
vals={r['path']:r for r in map(json.loads,(O/'data_values.jsonl').read_text(encoding='utf8').splitlines())}
groups=collections.defaultdict(list)
for r in rows:groups[(r['folder'],r['session'])].append(r)
runs=[]
for key,rs in sorted(groups.items()):
 rs.sort(key=lambda r:(int(r['epoch']),int(r['seg'])))
 assert len({(r['epoch'],r['seg']) for r in rs})==len(rs)
 run=None;prev=None
 for r in rs:
  e,s=int(r['epoch']),int(r['seg']);ordinal=e*20+s
  assert 0<=s<20 and 0<=e<2
  if prev is None or ordinal!=prev+1:
   run=dict(folder=key[0],session=key[1],run_index=sum(x['folder']==key[0] and x['session']==key[1] for x in runs),first_ordinal=ordinal,last_ordinal=ordinal,files=[]);runs.append(run)
  v=vals[r['path']]
  run['files'].append(dict(path=r['path'],epoch=e,seg=s,source_split=r['split'],recording_sample_start=ordinal*6144000,run_sample_start=len(run['files'])*6144000,samples=v['n'],all_zero=v['nonzero']==0,first_nonzero=v['first_nonzero']))
  run['last_ordinal']=ordinal;prev=ordinal
for run in runs:run['samples']=sum(r['samples'] for r in run['files'])
assert sum(len(r['files']) for r in runs)==3836
(O/'data_continuous_runs.json').write_text(json.dumps({'basis':facts,'recordings':len(groups),'continuous_runs':len(runs),'files':len(rows),'runs':runs},ensure_ascii=False,indent=2),encoding='utf8')
with (O/'data_continuous_runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['folder','session','run_index','first_ordinal','last_ordinal','files','samples']);w.writeheader()
 for r in runs:w.writerow({**{k:v for k,v in r.items() if k!='files'},'files':len(r['files'])})
print(json.dumps({'recordings':len(groups),'continuous_runs':len(runs),'files':len(rows)},ensure_ascii=False))
