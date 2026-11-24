from pathlib import Path
import json

W=Path(__file__).parent; O=W/'audit_drone_20261023'
p=O/'user_confirmations.json';d=json.loads(p.read_text(encoding='utf8'))
d['source_directory_split_status']='train/test are storage directories, not an accepted training/test split'
d['split_instruction_source']='用户于2026-10-24转述导师：数据由研究者自行划分'
d['split_policy']='Preserve original source paths for traceability; create research split independently before windowing.'
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
replacements={
'train的seg00–17、test的seg18–19来自同一录制键。没有证明它们样点重叠，但存在同次录制相关性，不能解释为独立采集泛化。':'用户于2026-10-24转述导师确认：data/train与data/test不是已划分好的训练/测试集，由研究者自行划分。现有目录恰好分别存放seg00–17、seg18–19，应一起纳入源数据索引；source_split只表示来源目录。后续划分需独立制定，同录制相关性仍须考虑。',
'不能用已有test反复调筛选和模型后仍称最终测试。':'现有test目录不视为最终测试集；应在明确研究划分后冻结测试范围，再进行筛选与模型选型。',
'跨原train/test边界的同录制不能据文件夹名称视为独立测试。':'用户已明确原train/test仅为存放目录；它们都进入同一源数据池，后续由研究者重新划分。',
}
for name in ['reports/实验室数据复核与改进方案_20261023.md','work/record_lab_confirmations_20261023.py']:
    p=W.parent/name;s=p.read_text(encoding='utf8')
    for a,b in replacements.items():s=s.replace(a,b)
    # Preserve the later split confirmation when this older helper is rerun.
    if name.endswith('.py'):
        old="(O/'user_confirmations.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf8')"
        new="if (O/'user_confirmations.json').exists():\n old_facts=json.loads((O/'user_confirmations.json').read_text(encoding='utf8'));facts={**old_facts,**facts}\n"+old
        if 'old_facts=' not in s:s=s.replace(old,new)
    p.write_text(s,encoding='utf8')
p=W/'deepseek-context.md';s=p.read_text(encoding='utf8')
note='\n\n2026-10-24用户进一步转述导师：data/train和data/test不是已划分好的数据集，应自行划分。source_split仅表示原存放目录；training_split仍unassigned。两目录全部纳入源数据池，尚未创建新的研究划分。\n'
if '2026-10-24用户进一步转述导师' not in s:p.write_text(s+note,encoding='utf8')
print('Updated source-directory split semantics; no new research split created.')
