from pathlib import Path
import json,csv,collections
O=Path(__file__).parent/'audit_drone_20261023';root=Path(__file__).parent.parent
assert (O/'values_complete.json').exists(),'Full numeric scan still running'
a=[json.loads(s) for s in (O/'data_values.jsonl').read_text(encoding='utf8').splitlines()]
idx=list(csv.DictReader((O/'data_index.csv').open(encoding='utf-8-sig')));unique=[r for r in idx if r['duplicate']=='False'];expected={r['path'] for r in unique}
assert len(a)==len(expected)==3836 and {r['path'] for r in a}==expected
zero=[r for r in a if r.get('nonzero')==0];partial=[r for r in a if r.get('nonzero',0)>0 and r.get('zero_blocks_61440')];err=[r for r in a if 'error' in r]
summary=dict(files=len(a),read_errors=err,all_zero_files=len(zero),nonzero_files_with_zero_blocks=len(partial),nonfinite_files=sum(r.get('finite') is False for r in a),integer_iq_files=sum(r.get('integer_iq') is True for r in a),all_dtype_complex64=all(r.get('dtype')=='complex64' for r in a),all_size_6144000=all(r.get('n')==6144000 for r in a))
(O/'numeric_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');(O/'zero_and_partial_files.json').write_text(json.dumps({'all_zero':zero,'partial_zero_blocks':partial},ensure_ascii=False,indent=2),encoding='utf8')
by={r['path']:r for r in a};rows=[]
for r in unique:
 v=by[r['path']];q=dict(r);q.update({k:v.get(k) for k in ['nonzero','first_nonzero','last_nonzero','finite','max_abs']});q['all_zero']=v.get('nonzero')==0;q['has_zero_block']=bool(v.get('zero_blocks_61440'));q['fs_status']='unknown';q['fc_status']='unknown';rows.append(q)
with (O/'data_unique_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=root/'reports/实验室数据复核与改进方案_20261023.md'
txt=report.read_text(encoding='utf8');txt+='\n\n## 8. 全量IQ数值检查最终结果\n\n'
txt+=f'对全部{len(a)}个去重data MAT完整解压并检查IQ，读取错误{len(err)}个，非有限值文件{summary["nonfinite_files"]}个。complex64与6,144,000点检查分别为{summary["all_dtype_complex64"]}/{summary["all_size_6144000"]}；I/Q均为整数值的文件{summary["integer_iq_files"]}个。副本由97组全文件SHA-256证明一致，不重复解压。\n\n'
txt+=f'发现**{len(zero)}个全零文件**，以及**{len(partial)}个仍含非零数据但存在连续全零块的文件**。连续块按61,440样点分块检查（尚未知fs，不标为毫秒）；不排除其余文件存在更短的零串。全零文件不能标为真实背景，原因仍未确认。\n\n'
txt+='|问题|文件|首个非零点（0基）|最后非零点（0基）|\n|---|---|---:|---:|\n'
for kind,rr in [('全零',zero),('部分全零块',partial)]:
 for r in rr:txt+=f'|{kind}|{r["path"]}|{r.get("first_nonzero")}|{r.get("last_nonzero")}|\n'
txt+='\n证据：data_values.jsonl记录每文件检查；numeric_summary.json汇总；zero_and_partial_files.json列出异常；data_unique_manifest.csv合并筛选标记。首末非零点只描述数值范围，不是无人机信号起止检测结果，不能直接当作目标标签。尚未对所有文件做射频语义标注或验证低信噪比检测性能。\n'
report.write_text(txt,encoding='utf8')
p=root/'work/deepseek-workflow.json';s=json.loads(p.read_text(encoding='utf8'));s.update(executionStatus='audit_completed_literature_proposal_delivered',currentTask='LAB_DRONE_AUDIT_20261023',nextTask='Confirm data acquisition metadata and discuss preprocessing proposal',nextTaskFile='reports/实验室数据复核与改进方案_20261023.md',nextTaskStatus='await_data_acquisition_metadata',reviewRequired=False);p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
p=root/'work/deepseek-context.md';s=p.read_text(encoding='utf8');s+=f'\n全量数值复核已完成：{len(a)}个去重MAT，{len(zero)}个全零文件，{len(partial)}个含整块零且仍有非零数据的文件，读取错误{len(err)}。详情见报告第8节；不推断补零原因。仅采集参数等外部事实仍待确认。\n';p.write_text(s,encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
