import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));from extract import NAMES

def load(n):return json.loads((O/n).read_text(encoding='utf-8'))
def f(v):return f'{v:.6g}'
def main():
 a=pd.read_csv(O/'bgA_index.csv');b=pd.read_csv(O/'bgB_index.csv',dtype={'nn':str});c=pd.read_csv(O/'B2440_index.csv');d=pd.read_csv(O/'A013_index.csv');m=pd.read_csv(O/'main_table.csv');t1=pd.read_csv(O/'appendix1_all.csv');t2=pd.read_csv(O/'appendix2_inband.csv');h=load('hash_verification.json');imp=load('implementation.json');maps=load('mapping.json');cc=load('feature_cache_check.json');white=load('white_check.json');lr=load('lr_oof.json')
 positive=bool((m[m.feature.isin(['ctl_abs_power_db','ctl_floor_db'])].auc>=.9).all())
 lines=['# '+('' if positive else '阳性对照未分开；')+'B1b 接收机 B 镜像背景与内容可比来源检查','实现：复制任务 F/I 的 Source、chain、kernel、ratio、process；未 import crop.py；滤波器及陷波表只读；全部为 CPU 处理，无网络训练。',f'复现检查：复制/原块长度 {imp["n_copy"]}/{imp["n_original"]}，最大绝对误差 {imp["max_abs"]}，最大逐点相对误差 {imp["max_point_relative"]} < 1e-5。', '映射推导：共轭后偏移 = −(A−2367.0588)，fc_in=2432.9412，故输出绝对频率 = 4800−A MHz。', '三频点验证：'+ '；'.join(f'{r["input_mhz"]}→{r["measured"]:.9f} MHz（误差 {r["error"]:.9g} MHz）' for r in maps)+'，均 ≤0.01 MHz。',f'SHA256：前后 {h["n_before"]}/{h["n_after"]} 个文件完全一致，新增/删除/变化 {len(h["changed"])}；详见 hash_before.json、hash_after.json、hash_verification.json。',f'bgA 缓存抽查 {cc["n"]} 片，19 特征最大绝对差 {cc["max_abs"]} ≤1e-9；无非有限值的丢弃/填充用于单特征 AUC。','安静规则：1024 点 Hann、hop=1024；STFT 帧按中心时刻归入 [j,j+1) ms，频率归入 [2405+k,2406+k) MHz，排除陷波±0.3 MHz；先在线性功率上分别平均，以全部20×50单元中位数为底噪，最大超出≤3 dB。',f'白噪声自检：{sum(r["quiet"] for r in white)}/20（{sum(r["quiet"] for r in white)/20:.0%}），最大单元超出范围 {min(r["max_excess_db"] for r in white):.4f}–{max(r["max_excess_db"] for r in white):.4f} dB。',f'bgA 安静 {int(a.quiet.sum())}/5138；bgB 安静 {int(b.quiet.sum())}/{len(b)}；B 每块输入时长 '+','.join(f'{x["nn"]}:{x["n_in"]/1e6:.5f}' for x in load('plan_bgB.json'))+' ×10 ms。']
 counts=b.groupby('nn').quiet.agg(['sum','count']);lines.append('各 NN 安静/总数：'+'；'.join(f'{nn} {int(r["sum"])}/{int(r["count"])}' for nn,r in counts.iterrows())+'；按块AUC跳过安静<3：'+','.join(nn for nn,r in counts.iterrows() if r['sum']<3)+'。')
 lines+=['主表：y=1 为 bgB；AUC=max(raw,1−raw)，方向按 raw；两侧数字为中位数 [p5,p95]；n0/n1='+f'{int(a.quiet.sum())}/{int(b.quiet.sum())}。','|特征|AUC|方向|bgA 中位数 [p5,p95]|bgB 中位数 [p5,p95]|按块AUC min–max|标记|','|---|---:|---|---|---|---|---|']
 for _,r in m.iterrows():lines.append(f'|{r.feature}|{r.auc:.5f}|{r.direction}|{f(r.A_median)} [{f(r.A_p5)},{f(r.A_p95)}]|{f(r.B_median)} [{f(r.B_p5)},{f(r.B_p95)}]|{r.block_min:.5f}–{r.block_max:.5f}|{r.mark}|')
 lines += [f'阳性对照：绝对功率 AUC={m.set_index("feature").loc["ctl_abs_power_db","auc"]:.6f}，底噪 AUC={m.set_index("feature").loc["ctl_floor_db","auc"]:.6f}；'+('均≥0.90。' if positive else '有项目<0.90。'),f'13候选特征逻辑回归：留一 NN 块、bgA 随机5折（seed=20261110），合并 OOF AUC={lr["auc"]:.6f}；C=1、balanced、仅训练折中位数填充与标准化；没有重新划分原数据集。', '各折测试 nA/nB：'+'；'.join(f'{r["held_nn"]}={r["A_test"]}/{r["B_test"]}' for r in lr['folds'])+'。',f'附表1：全部 bgA vs 全部 bgB（n0/n1={len(a)}/{len(b)}）；附表2：A013/F013 原指定19片 vs B@2440安静片（n0/n1={len(d)}/{int(c.quiet.sum())}）。',f'通带内补充筛查：A013/F013 在本次判据下 {int(d.quiet.sum())}/{len(d)} 安静，但按任务指定19片全部纳入附表2；B@2440 {int(c.quiet.sum())}/{len(c)} 安静。','|特征|附表1 AUC|方向|附表2 AUC|方向|','|---|---:|---|---:|---|']
 for i,r in t1.iterrows():s=t2.iloc[i];lines.append(f'|{r.feature}|{r.auc:.5f}|{r.direction}|{s.auc:.5f}|{s.direction}|')
 lines+=['完整主表及两附表（含两侧中位数、p5/p95、n0/n1）见 main_table.csv、appendix1_all.csv、appendix2_inband.csv；逐块结果见 auc_per_block.csv。','![图17](图17_接收机A与B镜像背景对比.png)','图17：左侧为每片去均值/功率归一化后的1 MHz平均谱（bgA固定随机200片、bgB全部安静片），各侧相对自身通带中位数及B−A；中间各2片时频图色标−3…25 dB；右侧4项箱线图（不画离群点）。']
 times={n:load(n)['seconds'] for n in ['generation_runtime.json','screen_runtime.json','analysis_runtime.json']};esize=sum(p.stat().st_size for p in Path('E:/fusionB_bgB_20261110').rglob('*') if p.is_file());osize=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());iq=sum(p.stat().st_size for p in Path('E:/fusionB_bgB_20261110').glob('*__bgB.npy'))
 lines += [f'耗时：生成（含前置哈希）{times["generation_runtime.json"]:.1f}s，安静筛选/特征 {times["screen_runtime.json"]:.1f}s，AUC/逻辑回归/图 {times["analysis_runtime.json"]:.1f}s，阶段合计 {sum(times.values()):.1f}s（不含编辑及最终哈希）。',f'磁盘：5个bgB IQ块 {iq/1e6:.3f} MB；E目录含复现块 {esize/1e6:.3f} MB；B1b目录交付前 {osize/1e6:.3f} MB（十进制）。','自行决定：s0001/s0002按NN顺序相接；输出先trim两端0.2ms，20ms从块内0点不重叠切片，尾部不足20ms丢弃；不额外导出重复IQ片段，片段由bgB_index.csv中的块路径/start/n_samples定位。','异常：首次用PowerShell管道生成路径常量时中文变为问号，已改为__file__推导绝对路径；未触及保护文件，无读盘错误或实现检查失败。','局限：bgB仅07-15一个会话、约0.5s且片段不独立，bgA来自61个录制；接收机与时间/地点会话混杂，无法从本对照分离；按块AUC范围不是独立会话置信区间。','频段位置：2345–2395MHz在A为fc−55…fc−5MHz、接近采样边缘，在B为fc−22…fc+28MHz、处于通带中部；模拟滤波响应差异亦可能贡献来源区分。','**事实陈述，不下结论**',f'主比较样本：bgA={int(a.quiet.sum())}，bgB={int(b.quiet.sum())}。',f'c20n_abs 中位数：bgA={m.set_index("feature").loc["c20n_abs","A_median"]:.9g}；bgB={m.set_index("feature").loc["c20n_abs","B_median"]:.9g}。',f'预定标签计数：接收机指纹={int((m.mark=="接收机指纹").sum())}；可疑={int((m.mark=="可疑").sum())}；未见接收机差异={int((m.mark=="未见接收机差异").sum())}。',f'逻辑回归合并OOF AUC={lr["auc"]:.6f}。',f'保护目录 SHA256 变化文件数={len(h["changed"])}。']
 # Coordinator authorized the common added LO notch after the original 0/20 failure.
 if int(b.quiet.sum())<10:lines[0]='# 样本不足；'+lines[0].removeprefix('# ')
 lines.insert(2,'协调者通过 Orca ask 授权修正：B本振2367.0588MHz镜像到2432.9412MHz；两侧每片IQ同加该点陷波（复制F的6阶Butterworth高通混频函数，width=50000Hz），谱及安静掩模同排除±0.3MHz；3dB阈值不变。原规则bgB=0/20。')
 lines.insert(3,'协调者核实000278/000360的8片：超标均在2432/2433MHz单元，其余单元最大1.01dB；本次逐片峰值定位见quiet_failure_cells.csv。')
 diag=pd.read_csv(O/'lo_cell_diagnostic.csv',dtype={'nn':str})
 lines.insert(4,'8片复核（按NN/片内起点排序，本振单元峰值/其余单元峰值，dB）：'+'；'.join(f'{r.nn}:{r.lo_cell_max_db:.3f}/{r.other_cell_max_db:.3f}' for _,r in diag.iterrows())+'；详见lo_cell_diagnostic.csv。')
 nsel=int(m.iloc[0].n0)
 lines=[s.replace('n0/n1='+f'{int(a.quiet.sum())}/{int(b.quiet.sum())}', 'n0/n1='+f'{nsel}/{int(b.quiet.sum())}') for s in lines]
 for i,s in enumerate(lines):
  if s.startswith('附表1：'):lines[i]=f'附表1：授权随机抽取bgA {nsel}片 vs 全部bgB {len(b)}片（未按安静筛选B）；附表2：原F链A013/F013指定19片 vs B@2440安静{int(c.quiet.sum())}片（两侧此参考均不加新陷波）。'
  if s.startswith('安静规则：'):lines[i]=s+' 新增2432.9412±0.3MHz排除。'
  if s.startswith('自行决定：'):lines[i]=s+' 按授权用seed=2026从bgA安静集随机抽1500片重算19特征（bgA_analysis_index.csv）；其余bgA仅重筛。'
  if s.startswith('主比较样本：'):lines[i]=f'主比较样本：bgA={nsel}（从全部安静{int(a.quiet.sum())}片抽样），bgB={int(b.quiet.sum())}。'
  if s.startswith('耗时：'):lines[i]=s+f' 自基线清单保存至报告生成墙钟耗时{time.time()-(O/"hash_before.json").stat().st_mtime:.1f}s（含原筛选、问答等待及一次计算调度重启）。'
  if s.startswith('局限：'):lines[i]=s+' 新增陷波按片双向滤波，片边缘瞬态可能影响高阶矩。'
  if s.startswith('bgA 安静 '):lines[i]=f'bgA 安静 {int(a.quiet.sum())}/5138；bgB 安静 {int(b.quiet.sum())}/{len(b)}；B 每NN输入99.960ms，共499.800ms；20ms不重叠片段总覆盖400ms。'
  if s.startswith('|特征|附表1 AUC|'):lines[i]=s+'附表3 AUC|方向|'
  if s=='|---|---:|---|---:|---|':lines[i]='|---|---:|---|---:|---|---:|---|'
 t3=pd.read_csv(O/'appendix3_no_notch.csv').set_index('feature')
 for i,s in enumerate(lines):
  if s.startswith('|') and len(s.split('|'))==7 and s.split('|')[1] in t3.index:
   r=t3.loc[s.split('|')[1]];lines[i]=s[:-1]+f'|{r.auc:.5f}|{r.direction}|'
 a3=pd.read_csv(O/'appendix3_bgA_index.csv');b3=pd.read_csv(O/'appendix3_bgB_index.csv')
 lines.insert(next(i for i,s in enumerate(lines) if s.startswith('|特征|附表1')),f'附表3：不加新陷波、安静判据仅额外排除2432/2433MHz两整单元；n0/n1={int(a3.quiet.sum())}/{int(b3.quiet.sum())}；特征仍为原B1完整实现（含本振尖峰），完整分位数见appendix3_no_notch.csv。')
 lines=[s for s in lines if not s.startswith('完整主表及两附表')]
 lines[1]+=' 完整表见main_table.csv、appendix1_all.csv、appendix2_inband.csv；逐块见auc_per_block.csv。'
 lines.insert(1,'')
 for header in ['|特征|AUC|','|特征|附表1 AUC|']:
  lines.insert(next(i for i,s in enumerate(lines) if s.startswith(header)),'')
 assert len(lines)<=80,len(lines);(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('REPORT_LINES',len(lines))
if __name__=='__main__':main()
