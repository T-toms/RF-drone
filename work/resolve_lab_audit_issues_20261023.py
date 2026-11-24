"""Build usable source/interval indices from existing audit evidence. Never edit IQ."""
from pathlib import Path
import csv, json, collections, hashlib
from drone_audit_support import replace_markdown_section

W=Path(__file__).parent; O=W/'audit_drone_20261023'; ROOT=Path('G:/drone')
def read(name): return json.loads((O/name).read_text(encoding='utf8'))
def write(name,obj): (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def csv_write(name,rows):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

index=list(csv.DictReader((O/'data_index.csv').open(encoding='utf-8-sig')))
by_path={r['path']:r for r in index}; assert len(by_path)==len(index)==3933
values={v['path']:v for v in map(json.loads,(O/'data_values.jsonl').read_text(encoding='utf8').splitlines())}
assert set(values)=={r['path'] for r in index if r['duplicate']=='False'}
canonical={p:p for p in values}
for pair in read('duplicate_hashes.json'):
    assert pair['equal'] and len({x[1] for x in pair['files']})==1
    originals=[p for p,h in pair['files'] if by_path[p]['duplicate']=='False'];assert len(originals)==1
    for p,h in pair['files']:canonical[p]=originals[0]
assert set(canonical)==set(by_path)
data=[]
for r in index:
    p=r['path'];v=values[canonical[p]];assert v['finite'] and v['n']==6144000
    reasons=[];is_dup=p!=canonical[p];allzero=v['nonzero']==0
    if is_dup:reasons.append('verified_duplicate')
    if allzero:reasons.append('all_zero')
    trim=0
    if not allzero and v['zero_blocks_61440']:
        trim=v['first_nonzero'];assert all((b+1)*61440<=trim for b in v['zero_blocks_61440'])
    data.append(dict(path=p,canonical_path=canonical[p],folder=r['folder'],session=r['session'],epoch=int(r['epoch']),seg=int(r['seg']),source_split=r['split'],duplicate=is_dup,all_zero=allzero,analysis_eligible=not reasons,exclusion_reason='|'.join(reasons),start_sample=None if allzero else trim,end_sample_exclusive=None if allzero else v['n'],removed_zero_prefix_samples=trim,sample_rate_hz=None,exact_receiver_center_hz=None,training_split='unassigned'))
csv_write('data_preprocessing_manifest.csv',data);lookup={r['path']:r for r in data}
valid_runs=[]
for run in read('data_continuous_runs.json')['runs']:
    active=None
    for f in run['files']:
        r=lookup[f['path']]
        if not r['analysis_eligible']:active=None;continue
        start=f['recording_sample_start']+r['start_sample'];end=f['recording_sample_start']+r['end_sample_exclusive']
        if active is None or active['end_sample_exclusive']!=start:
            active=dict(folder=run['folder'],session=run['session'],original_run_index=run['run_index'],start_sample=start,end_sample_exclusive=start,parts=[]);valid_runs.append(active)
        active['parts'].append(dict(path=f['path'],start_sample=r['start_sample'],end_sample_exclusive=r['end_sample_exclusive'],recording_sample_start=start,source_split=r['source_split']))
        active['end_sample_exclusive']=end
for r in valid_runs:
    r['samples']=r['end_sample_exclusive']-r['start_sample']
    assert r['samples']==sum(f['end_sample_exclusive']-f['start_sample'] for f in r['parts'])
write('data_usable_runs.json',{'coordinate_system':'0-based, half-open sample intervals; original recording offsets retained','purpose':'numerically usable intervals for analysis, not UAV activity labels or final training splits','sample_rate_hz':None,'runs':valid_runs})

hashes=[json.loads(s) for s in (O/'raw_full_hash.jsonl').read_text(encoding='utf8').splitlines()]
meta={str((ROOT/x['path']).with_suffix('.bin')):x['metadata'] for x in read('raw_json_complete.json')}
assert set(meta)=={r['path'] for r in hashes}
groups=collections.defaultdict(list)
for r in hashes:groups[r['actual']].append(r['path'])
canonical_raw={p:min(ps) for ps in groups.values() for p in ps}
raw=[]
for h in hashes:
    p=h['path'];m=meta[p];c=m['capture'];reasons=[];dup=canonical_raw[p]!=p;zero=h['zero_iq_samples']*4==h['bytes']
    if dup:reasons.append('verified_duplicate')
    if not h['matches']:reasons.append('sha256_mismatch')
    if zero:reasons.append('all_zero')
    duration=c['sample_count']/c['sample_rate_hz']
    raw.append(dict(path=p,canonical_path=canonical_raw[p],session=m['session']['id'],label=m['declared']['class'],models='|'.join(m['declared']['model']),duplicate=dup,analysis_eligible=not reasons,exclusion_reason='|'.join(reasons),sample_rate_hz=c['sample_rate_hz'],center_freq_hz=c['center_freq_hz'],samples=c['sample_count'],duration_ms=duration*1000,supports_10ms=not reasons and duration>=.01,supports_20ms=not reasons and duration>=.02,training_split='unassigned'))
csv_write('raw_preprocessing_manifest.csv',raw);rl={r['path']:r for r in raw}
pairs=[]
for x in read('raw_pairing_config_audit.json')['details']:
    p=canonical_raw[str((ROOT/x['drone']).with_suffix('.bin'))]
    if not rl[p]['analysis_eligible']:continue
    candidates=sorted({canonical_raw[str((ROOT/y).with_suffix('.bin'))] for y in x['also_gain_agc']})
    candidates=[y for y in candidates if rl[y]['analysis_eligible']]
    assert candidates
    pairs.append({'drone':p,'backgrounds':candidates})
assert len({x['drone'] for x in pairs})==len(pairs)
write('raw_usable_background_pairs.json',{'matching_fields':['sample_rate_hz','center_freq_hz','usable_bw_hz','gain_db','agc'],'not_guaranteed':['simultaneous capture','same physical receiver','Wi-Fi/Bluetooth ground truth'],'pairs':pairs})
summary=dict(data_source_files=len(data),data_duplicates=sum(r['duplicate'] for r in data),data_all_zero_unique=sum(r['all_zero'] and not r['duplicate'] for r in data),data_eligible_files=sum(r['analysis_eligible'] for r in data),data_trimmed_prefix_files=sum(r['analysis_eligible'] and r['removed_zero_prefix_samples']>0 for r in data),data_usable_runs=len(valid_runs),data_valid_samples=sum(r['samples'] for r in valid_runs),raw_source_files=len(raw),raw_duplicates=sum(r['duplicate'] for r in raw),raw_eligible_files=sum(r['analysis_eligible'] for r in raw),raw_eligible_drone_files=sum(r['analysis_eligible'] and r['label']=='drone' for r in raw),raw_eligible_background_files=sum(r['analysis_eligible'] and r['label']=='ambient' for r in raw),raw_eligible_under_10ms=sum(r['analysis_eligible'] and not r['supports_10ms'] for r in raw),usable_paired_drone_files=len(pairs),usable_paired_background_files=len({p for x in pairs for p in x['backgrounds']}))
assert summary['data_eligible_files']==3821 and summary['data_trimmed_prefix_files']==25
assert summary['raw_eligible_files']==879 and summary['usable_paired_drone_files']==214
write('issue_resolution_summary.json',summary)
evidence=['data_index.csv','data_values.jsonl','duplicate_hashes.json','data_continuous_runs.json','raw_full_hash.jsonl','raw_json_complete.json','raw_pairing_config_audit.json','user_confirmations.json']
write('preprocessing_evidence_hashes.json',{name:hashlib.sha256((O/name).read_bytes()).hexdigest() for name in evidence})
section=f'''## 10. 已知问题处理与复查

复查日期：2026-10-24。14项索引、配对和报告重跑检查通过。

已将分散的异常规则合并为可执行的预处理索引，原始IQ未移动、删除或改写。这些索引只解决已验证的数值完整性与重复问题，不代表已经识别出无人机有效信号，也不指定训练/验证/测试划分。

|数据|处理|结果|
|---|---|---|
|data|97个确认副本不重复接入；15个全零原件不进入分析|3821个文件保留可用区间|
|data|25个已确认长零前缀文件从首个非零点起保留，坐标仍对应原始录制|生成{len(valid_runs)}个数值可用连续区间，缺段处不连接|
|raw|10个确认副本不重复接入；1个全零且哈希不符文件排除|879个文件，其中214个drone、665个ambient|
|raw配对|将完整性与去重规则应用到配对关系|214个drone对应45个不同背景文件|
|raw短片段|30个5 ms背景显式标记不能生成真实10/20 ms窗口|仍可用于符合时长的分析|

文件：data_preprocessing_manifest.csv（含全部3933条源记录与排除原因）、data_usable_runs.json（可用样点区间）、raw_preprocessing_manifest.csv（890条源记录与规则）、raw_usable_background_pairs.json（可用配对）、issue_resolution_summary.json（计数）。旧data_continuous_runs.json保留原始录制结构，包含零段用于追溯；后续分析应读取data_usable_runs.json，不能把旧结构索引当作已经剔除异常的数据。

start_sample/end_sample_exclusive采用0基半开区间。仅对已检出的长零前缀截取；其他文件中偶发的数值零不自动当坏数据。analysis_eligible表示通过本轮数值/去重规则，不是最终训练资格；所有training_split保持unassigned。data采样率、精确接收中心频率、combination与异常原因仍待采集信息；没有用猜测填充。

报告更新脚本已改为只替换对应章节，避免重跑时重复插入文字或删除后续补充。复查结果另见issue_resolution_verification.json。证据文件摘要见preprocessing_evidence_hashes.json；本轮复用了已完成的全量读取结果，没有重新读取整个数据集。
'''
p=W.parent/'reports/实验室数据复核与改进方案_20261023.md';p.write_text(replace_markdown_section(p.read_text(encoding='utf8'),'## 10. 已知问题处理与复查',section),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
