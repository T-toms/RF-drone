"""Consolidate measured evidence; never turn failed checks into passes."""
import importlib.util,datetime,collections
from pathlib import Path
s=importlib.util.spec_from_file_location('crop',Path(__file__).with_name('crop.py'));c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def main():
 O=c.O;v=c.read(O/'full_verification.json');v.update(c.read(O/'index_verification.json'));before=c.read(O/'source_before.json');after=[(str(p.relative_to(c.R)),p.stat().st_size,p.stat().st_mtime_ns) for p in c.R.rglob('*') if p.is_file()];c.dump(O/'source_after.json',after);aa=[list(x) for x in after];v['source_unchanged']=dict(status='pass' if sorted(before)==sorted(aa) else 'fail',files_before=len(before),files_after=len(after),latest_mtime_before=max(r[2] for r in before),latest_mtime_after=max(r[2] for r in after))
 # Re-read actual delivered CSVs independently, across all lengths.
 block_rows=c.rows(O/'blocks.csv');lengths={r['block_file']:int(r['n_samples']) for r in block_rows};block_groups={r['block_file']:r['group_id'] for r in block_rows};assigned={r['group_id']:r['split'] for r in c.rows(O/'group_split.csv')};group_splits=collections.defaultdict(set);block_splits=collections.defaultdict(set);bad=0;n=0;flagbad=0;mappingbad=0
 for L in [10,20,30,40,50]:
  for r in c.rows(O/f'index_L{L}.csv'):
   n+=1;group_splits[r['group_id']].add(r['split']);block_splits[r['block_file']].add(r['split']);start,ns=int(r['start']),int(r['n_samples']);bad+=int(start<0 or start+ns>lengths[r['block_file']] or ns!=round(c.FS[r['band']]*L/1000) or start%ns!=0);flags=[r[k]=='True' for k in ['lack_feature','primary_weak','power_shift','floor_raised','interference']];flagbad+=int((r['keep']=='True')!=(not any(flags)) or (r['label']=='background' and (any(flags) or r['keep']!='True')));mappingbad+=int(r['group_id']!=block_groups[r['block_file']] or r['split']!=assigned[r['group_id']])
 leaks=sum(len(s)!=1 for s in group_splits.values())+sum(len(s)!=1 for s in block_splits.values())+mappingbad;v['csv_leakage']=dict(status='pass' if not leaks else 'fail',violations=leaks,groups=len(group_splits),blocks=len(block_splits),rows=n);v['csv_boundaries']=dict(status='pass' if not bad else 'fail',violations=bad,rows=n);v['screening_flags']=dict(status='pass' if not flagbad else 'fail',violations=flagbad,rows=n);c.dump(O/'verification.json',v)
 bs=c.rows(O/'blocks.csv');gs=c.rows(O/'group_split.csv');counts=c.rows(O/'counts.csv');rates=c.rows(O/'retention.csv');excluded=c.rows(O/'excluded_files.csv');cfg=c.read(O/'config.json');size=sum(Path(r['block_file']).stat().st_size for r in bs);started=datetime.datetime.fromtimestamp((O/'source_before.json').stat().st_mtime);hours=(datetime.datetime.now()-started).total_seconds()/3600
 lines=['# 固定频段截取、切片、筛选及同源划分结果',f'完成 {len(bs)} 块、{len(gs)} 个同源组；IQ 输出 E:/drone_lab_crop_20261107，complex64，无归一化；未训练、未装包、未提交。',f'实际 IQ 占用 {size/1e9:.3f} GB；E 盘剩余 {c.shutil.disk_usage("E:/").free/1e9:.2f} GB；总耗时约 {hours:.2f} 小时（含冒烟、验收）。','输入路径实际为正斜杠，已兼容反斜杠；data model 由数字 code 映射，v3 背景 model=none；v3 fc 使用文件名。']
 inc=collections.Counter((r['dataset'],r['band'],r['label']) for r in bs);lines.append('纳入块数：'+'；'.join('/'.join(k)+'='+str(n) for k,n in sorted(inc.items())))
 paths=collections.defaultdict(set)
 for r in bs:paths[r['dataset']].update(r['source_paths'].split(';'))
 lines.append('纳入原始 MAT 文件数：'+'；'.join(f'{ds}={len(pp)}' for ds,pp in sorted(paths.items()))+'；逐块纳入清单与源路径见 blocks.csv。')
 exc=collections.Counter((str(r['fc_filename_mhz']),r['task_f_reason']) for r in excluded);c.write(O/'excluded_frequencies.csv',[dict(fc_mhz=k[0],reason=k[1],n_files=n) for k,n in sorted(exc.items())]);reasons=collections.Counter(r['task_f_reason'] for r in excluded);major=[2367.06,2500.,2514.59,2589.19,2600.,2663.78,5760.,5787.5,5810.,5881.56];nmajor={f:sum(float(r['fc_filename_mhz'])==f for r in excluded) for f in major};lines.append(f"排除 v3 文件 {len(excluded)} 个（源表排除{reasons['not_included']}、完整频段不覆盖{reasons['crop_band_not_covered']}）；主要中心 MHz/文件数："+'；'.join(f'{f:g}/{n}' for f,n in nmajor.items())+f"；其他扫频/会话{len(excluded)-sum(nmajor.values())}个；逐文件/频率明细见 excluded_files.csv、excluded_frequencies.csv。")
 lines.append('data 只用 include=True 的 100 个有效 run（96 次录制）；原始无效/重复文件不纳入；008/F008 共8次录制目录省略带宽 M 字符，沿用任务 A 解析为 10M/20M/40M，逐录制见 metadata_corrections.csv。')
 for case in v['v3_contiguity']['cases']:lines.append(f"连续性 {case['case']}：{case['status']}，两段区间 "+'、'.join(f"[{x['start']},{x['end']})" for x in case['checks'])+f"，最低首尾相关 {min(min(x['head_corr'],x['tail_corr']) for x in case['checks']):.8f}。")
 lines.append('已验证 250M 类按同 NN 拼接；扫频复核相接后同 NN 拼接；noise_adrv、缺配对文件单独成块；不同 NN 不拼接。')
 for b,p in cfg['filters'].items():lines.append(f"终端 FIR {b}：{p['taps']} taps，通带波动 {p['pass_ripple_db']:.6f} dB，阻带衰减 {p['stop_atten_db']:.2f} dB；两端各裁 {p['trim_samples']} 点（0.2 ms）。")
 lines.append('重采样使用显式 100 dB Kaiser FIR；所有来源共用对应频段的终端 FIR（filters.npz）；分段 50 ms、双侧 2 ms 重叠，避免内存溢出。')
 lines.append('统一陷波（频率 MHz / 半宽 kHz）：'+'；'.join(f"{r['band']} {r['freq_mhz']:.8f}/{r['half_width_khz']:g}" for r in cfg['notches'])+'；完整来源见 notch_table.csv。')
 lines.append('lab_preproc.notch 的 width 参数内部翻倍，因此调用传 half_width/2，确保最终半宽与陷波表一致。')
 lines.append('1 ms cell：2.4G NFFT=512（120 kHz/bin），5.8G NFFT=1024（97.65625 kHz/bin），Hann、不重叠、不折叠；5.8G 每 cell 用 97 帧，末 672 点不入谱。')
 lines.append('筛选按新 IQ 计算 recording 参考底噪、占用≥5% 子带，间隔≤2有效 bin 合并；固定陷波区作为环境 bin；v3b 五项阈值不变；背景全保留，仅标记不删除。')
 lines+=['|L ms|总片段|keep 后|保留率|','|---|---:|---:|---:|']
 for L,d in v['counts']['per_L'].items():lines.append(f"|{L}|{d['n']}|{d['keep']}|{d['keep']/d['n']:.2%}|")
 lines+=['各 split × 来源 × 频段 × 标签片段数（格式：总数/keep）：','|L|split|data 2.4 无人机|data 5.8 无人机|v3 2.4 无人机|v3 2.4 背景|v3 5.8 无人机|v3 5.8 背景|','|---|---|---|---|---|---|---|---|']
 for L in [10,20,30,40,50]:
  for split in ['train','val','test']:
   nums=[]
   for ds,b,label in [('data','2.4GHz','drone'),('data','5.8GHz','drone'),('v3','2.4GHz','drone'),('v3','2.4GHz','background'),('v3','5.8GHz','drone'),('v3','5.8GHz','background')]:
    r=next((r for r in counts if int(r['L'])==L and r['split']==split and r['dataset']==ds and r['band']==b and r['label']==label),None);nums.append(f"{r['n']}/{r['keep']}" if r else '0/0')
   lines.append('|'+ '|'.join([str(L),split]+nums)+'|')
 lines.append('秒数与 keep 秒数见 counts.csv；逐 model × split 数量见 model_counts.csv；L × band × state 保留率见 retention.csv。')
 for L in [10,20,30,40,50]:lines.append(f'{L}ms 保留率：'+'；'.join(f"{r['band']}/{r['state']}={float(r['retention']):.2%}" for r in rates if int(r['L'])==L))
 short=c.rows(O/'scarce_groups.csv') if (O/'scarce_groups.csv').exists() else [];lines.append('划分种子2026，同一 group 的所有 L 共用一个 split；优先保证 model 覆盖，按时长贪心。组数不足层/model：'+'；'.join(f"{r['stratum']}/{r['model']}={r['n_groups']}组" for r in short)+'；详情 group_split.csv。')
 lines+=['|来源/频段/标签|train 组数/秒/占比|val 组数/秒/占比|test 组数/秒/占比|','|---|---|---|---|']
 for key in sorted({(g['dataset'],g['band'],g['label']) for g in gs}):
  group=[g for g in gs if (g['dataset'],g['band'],g['label'])==key];total=sum(float(g['n_seconds']) for g in group);values=[]
  for split in ['train','val','test']:
   subset=[g for g in group if g['split']==split];seconds=sum(float(g['n_seconds']) for g in subset);values.append(f'{len(subset)}/{seconds:.3f}/{seconds/total:.1%}')
  lines.append('|'+ '|'.join(['/'.join(key)]+values)+'|')
 for band in c.FS:
  pp=[p for p in v['floor_shape']['pairs'] if p['band']==band];worst=max(pp,key=lambda r:r['max_difference_db']);lines.append(f"底噪形状 {band}：最大差 {worst['max_difference_db']:.3f} dB @ {worst['frequency_mhz']:.6f} MHz（{worst['source_a']} / {worst['source_b']}）；未自行修正，全部两两差值见 verification.json，图见 floor_shape_{band}.png。")
 lines.append('验证状态：'+'；'.join(k+'='+r['status'] for k,r in v.items()))
 lines.append(f"重算最大相对误差 {max(r['max_relative_error'] for r in v['recompute']['samples']):.3g}；陷波残留最大 {max(r['residual_db'] for r in v['notch']['samples']):.3f} dB，外侧功率变化最大 {max(r['max_outside_delta_db'] for r in v['notch']['samples']):.6f} dB。")
 lines.append('异常：底噪形状差异超过1 dB，包含低功率 cell 仍保留的真实信号影响，不将其强行平坦化；其他失败项如有，以 verification.json 数值为准。')
 lines+=['','**已知残余风险**：data/ 的接收机只有无人机样本、没有背景样本，此来源与标签相关性无法由本次预处理消除。']
 # Keep Markdown tables separated from prose while respecting the 70-line limit.
 lines[1]+=' '+lines.pop(2)
 i=next(i for i,line in enumerate(lines) if line.startswith('lab_preproc.notch'));lines[i-1]+=' '+lines.pop(i)
 lines=[line for line in lines if not line.startswith('各 split × 来源 ×')]
 lines=[line.replace('|L|split|','|L（各列总数/keep）|split|') for line in lines]
 rendered=[]
 for i,line in enumerate(lines):
  if line.startswith('|L（') and rendered and rendered[-1]!='':rendered.append('')
  if line.startswith('|') and i and not lines[i-1].startswith('|') and rendered[-1]!='':rendered.append('')
  if not line.startswith('|') and i and lines[i-1].startswith('|') and line!='':rendered.append('')
  rendered.append(line)
  if i==0:rendered.append('')
 assert len(rendered)<=70,len(rendered);(O/'REPORT.md').write_text('\n'.join(rendered)+'\n',encoding='utf-8');print('REPORT_COMPLETE',len(rendered),'lines',flush=True)
if __name__=='__main__':main()
