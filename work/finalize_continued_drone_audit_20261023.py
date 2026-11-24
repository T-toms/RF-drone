from pathlib import Path
import json,csv,collections,datetime
from drone_audit_support import replace_markdown_section

O=Path(__file__).parent/'audit_drone_20261023'
ROOT=Path(__file__).parent.parent
rows=[json.loads(x) for x in (O/'raw_full_hash.jsonl').read_text(encoding='utf8').splitlines()]
assert len(rows)==890 and len({r['path'] for r in rows})==890
assert not any('error' in r for r in rows)
mat=json.loads((O/'raw_mat_full_comparison.json').read_text(encoding='utf8'))
assert len(mat)==13
png=json.loads((O/'png_full_decode.json').read_text(encoding='utf8'))
assert len(png)==902
meta=json.loads((O/'raw_json_complete.json').read_text(encoding='utf8'))
by_path={str((Path('G:/drone')/x['path']).with_suffix('.bin')):x['metadata'] for x in meta}
assert set(by_path)=={r['path'] for r in rows}
bad=[r for r in rows if not r['matches']]
zeros=[r for r in rows if r['zero_iq_samples']*4==r['bytes']]
rails=[r for r in rows if r['rail_components']]
groups=collections.defaultdict(list)
for r in rows:groups[r['actual']].append(r['path'])
duplicate_groups=[v for v in groups.values() if len(v)>1]
summary={'files':len(rows),'bytes_read':sum(r['bytes'] for r in rows),
 'sha256_matches':len(rows)-len(bad),'sha256_mismatches':bad,
 'all_zero_files':zeros,'int16_rail_files':len(rails),
 'int16_rail_components':sum(r['rail_components'] for r in rows),
 'content_unique_hashes':len(groups),'content_duplicate_groups':duplicate_groups,
 'mat_files':len(mat),'mat_all_equal':all(r['all_samples_equal_after_div32768'] for r in mat),
 'mat_samples_compared':sum(r['samples'] for r in mat),
 'png_decoded_files':len(png),'png_decode_passed':sum(r['verified'] for r in png),
 'completed_at':datetime.datetime.now().isoformat(timespec='seconds')}
(O/'continued_final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
with (O/'raw_integrity_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['path','session_id','class','models','sample_rate_hz','center_freq_hz','samples','sha256_match','all_zero','zero_fraction','int16_rail_components','integrity_eligible','exclusion_reason']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rows:
  m=by_path[r['path']];z=r['zero_iq_samples']*4==r['bytes'];reason=[]
  if not r['matches']:reason.append('sha256_mismatch')
  if z:reason.append('all_zero')
  w.writerow(dict(path=r['path'],session_id=m['session']['id'],**{'class':m['declared']['class']},models='|'.join(m['declared']['model']),sample_rate_hz=m['capture']['sample_rate_hz'],center_freq_hz=m['capture']['center_freq_hz'],samples=r['bytes']//4,sha256_match=r['matches'],all_zero=z,zero_fraction=r['zero_iq_samples']/(r['bytes']//4),int16_rail_components=r['rail_components'],integrity_eligible=not reason,exclusion_reason='|'.join(reason)))

report=ROOT/'reports/实验室数据复核与改进方案_20261023.md'
s=report.read_text(encoding='utf8')
s=s.replace('902张PNG均成功打开图像头；','902张PNG均已通过Pillow校验与完整像素解码（png_full_decode.json）；' if all(r['verified'] for r in png) else '902张PNG已尝试完整解码，异常见png_full_decode.json；')
if '续查另目视核对了全零BIN的同名PNG' not in s:
 s=s.replace('未把902张逐张人工标注','续查另目视核对了全零BIN的同名PNG（见9.4节）；未把902张逐张人工标注')
s=s.replace('去除这些副本的逻辑索引有3,836条；原文件未删除。','去除这些副本的逻辑索引有3,836条；此数量不表示已证明剩余文件的内容两两不同，尚未对所有不同名称MAT开展全局内容哈希查重。原文件未删除。')
s=s.replace('2023；续查已核对采集格式与参数正文（9.3节），尚未下载数据。','2023年预印本，正式期刊版本发表于2024年IEEE Communications Magazine 62(5):50–56（[IEEE目录](https://www.comsoc.org/system/files/2024-06/Publications_Contents_Digest_2024_May.pdf)）；本轮方法核对依据所链接预印本正文（9.3节），尚未下载数据。')
if '作者提供的[CodeOcean代码入口]' not in s:
 s=s.replace('尚未实际下载或确认当前下载权限/许可证。','尚未实际下载或确认当前下载权限/许可证。作者提供的[CodeOcean代码入口](https://codeocean.com/capsule/5227169/tree)本轮返回403，不能据此认定数据文件或代码已取得。该采集在2.44 GHz，不能直接作为5.8 GHz实测背景的替代。')
s=s.replace('所有BIN头尾各读取最多4096字节成功；不等价于所有文件全字节校验成功。','初查所有BIN头尾读取成功；续查已完成全部890个BIN的全字节SHA-256及零值统计，发现的问题见9.4节。')
s=s.replace('这里只证明抽查块一致，没有声称13个MAT全部样点逐点比较完成。','续查已进一步完成13个MAT全部样点与同名BIN逐点比较，最终结果见9.4节。')
s=s.replace('其映射内容未完整解码，不当作原始背景IQ，也不推断缓存生成算法。','续查已用MATLAB解码映射内容，详见9.1节；不当作原始背景IQ，也不推断缓存生成算法。')
s=s.replace('其余409个没有完整重算哈希，因此只能说本次局部读未复现错误，不能宣称整目录无坏块。全量BIN哈希不在本轮已完成核验范围。','续查已将本目录全部410个文件纳入完整哈希，结果见9.4节。读取与哈希只能说明本轮文件内容核验结果，不是对整个存储介质健康状况的保证。')
s=s.replace('全量枚举与元数据核查不代表对全部图像人工标注，也不代表对全部BIN全字节哈希。','本次已完成全部BIN全字节哈希（9.4节）；仍未对全部图像人工标注。')
s=s.replace('作者摘要明确STFT与自相关，未逐页审读全文。','续查已读PDF的IV-B及V等相关正文，见9.3节。')
s=s.replace('已核对作者摘要中的分阶段检测、干扰与特征方法，不外推其准确率。','续查已核对原文方法与采集正文，适用边界见9.3节，不外推其准确率。')
s=s.replace('已核对摘要。数据发布方：','续查已核对采集正文，适用边界见9.3节。数据发布方：')
s=s.replace('已核对作者摘要，尚未下载数据。','续查已核对采集格式与参数正文（9.3节），尚未下载数据。')
section='''
### 9.4 全部raw BIN完整校验与13个MAT逐点比对

全部890个BIN均完整读完，共78,750,000,000字节；读取异常0。SHA-256与各自JSON一致{matches}个，不一致{badcount}个。这里核对的是现有BIN与现有JSON是否一致，不代表JSON本身具有外部可信签名。

全零BIN共{zerocount}个。异常清单如下：

|文件|实测SHA-256|JSON声明SHA-256|数值检查|
|---|---|---|---|
{table}

其中mavic3classic的000004文件：120,000,000字节即30,000,000个复数采样点全部为零。独立PowerShell Get-FileHash重算与Python结果相同。同名PNG仍显示非均匀时频结构，与当前全零BIN的内容不相符；PNG生成时间、使用的源版本和不一致原因未确定。该文件应在训练候选清单中隔离，保留原件供采集者/备份核对，不能把它归为真实静默或背景，也不能从PNG恢复原始相位。

按实际完整哈希统计有{unique}种不同BIN内容、{dupgroups}组相同内容；完整组表在continued_final_summary.json。历史报错的125 MS/s目录本轮完整读取{labn}个，其中{labmatch}个与JSON哈希一致；这不等价于硬盘硬件健康检查。

13个IQ MAT共{matpoints:,}个复数样点已全量比对，同名BIN按小端int16交织读取后除以32768，全部样点实部、虚部精确相等：{matequal}。因此这13个MAT是同名BIN的转换副本，不能增加独立采集数。

额外统计有{railfiles}个文件含int16数值端点（−32768或32767），共{railcount:,}个I/Q分量。端点计数是数字范围检查，不能单独证明或排除模拟前端饱和。零值比例也不是无线静默比例。

证据：raw_full_hash.jsonl、raw_full_hash_summary.json、raw_mat_full_comparison.json、continued_final_summary.json、raw_integrity_manifest.csv。manifest中的integrity_eligible只表示通过本轮哈希和非全零检查；未包含信号语义、标签可信度、重复去重和训练划分，不能直接作为最终训练清单。所有原始文件保持只读。

### 9.5 仍无法从当前文件确定的事项

导师已确认data以文件名为准、信号连续可拼接、20M表示20 MHz信号带宽（见9.0节）。仍未知的是data采样率、精确接收中心频率/有效带宽、combination语义、零前缀原因及缺失机型是否补采；需要对应采集配置或采集者说明。带宽不是采样率，不能用20M换算10/20 ms样点数。

本轮完成的是原始资料审计及文献适配复核，尚未执行全机型的长时频谱语义分析、子带切取、筛窗阈值选型或模型训练。可先在raw参数完整且完整性通过的文件上验证处理方法；data正式10/20 ms实验必须先补齐采样参数。

进一步的实验设计判断：如果正样本全部来自data、负样本全部来自raw，类别就与采集批次重合，模型可能依赖接收链、增益或数值尺度差异完成分类。统一数据类型、归一化和重采样不能保证消除这些差异。应优先获得与主数据同配置的环境背景；raw内部paired_bg可用于处理链验证，但不能据此替代主数据的跨条件评价。
'''.format(matches=len(rows)-len(bad),badcount=len(bad),zerocount=len(zeros),table='\n'.join('|`'+r['path']+'`|`'+r['actual']+'`|`'+r['expected']+'`|'+('全部为零' if r in zeros else '并非全部为零')+'|' for r in bad),unique=len(groups),dupgroups=len(duplicate_groups),labn=sum(by_path[r['path']]['capture']['sample_rate_hz']==125000000 for r in rows),labmatch=sum(r['matches'] and by_path[r['path']]['capture']['sample_rate_hz']==125000000 for r in rows),matpoints=sum(r['samples'] for r in mat),matequal='是' if summary['mat_all_equal'] else '否，见逐文件表',railfiles=len(rails),railcount=summary['int16_rail_components'])
first, second = section.strip().split('\n### 9.5 ', 1)
for block in (first, '### 9.5 ' + second):
 s=replace_markdown_section(s,block.splitlines()[0],block)
report.write_text(s,encoding='utf8')
state=ROOT/'work/deepseek-workflow.json';d=json.loads(state.read_text(encoding='utf-8-sig'));d.update(executionStatus='continued_audit_complete',nextTaskStatus='await_data_acquisition_metadata',updatedAt=datetime.datetime.now().astimezone().isoformat(),currentTask='LAB_DRONE_AUDIT_20261023',enabled=False,automaticWakeEnabled=False)
state.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
p=ROOT/'work/deepseek-context.md';s=p.read_text(encoding='utf8');marker='\n\n### 2026-10-23 续查完成'
s=replace_markdown_section(s,marker.strip(),marker+'\n\n全部890个raw BIN完整读取并核对JSON哈希；'+str(len(bad))+'个不一致，'+str(len(zeros))+'个全零。13个IQ MAT共390M复数点与BIN/32768全量比对已完成。缓存MATLAB解码仅含2440000000键与1024点向量，无data采样参数。25次受零前缀影响录制均为5.8GHz/epoch0。文献适配复核及剩余边界见报告第9节；未训练、未修改原始数据，继续保持DSH与定时唤醒关闭。\n')
p.write_text(s,encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
