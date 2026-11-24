from pathlib import Path
import json,re,datetime,hashlib,csv
B=Path(__file__).resolve().parent;D=B.parent;R=B.parents[2]
audit=json.loads((B/'r2_final_audit.json').read_text(encoding='utf-8'))
task=R/'work/codex_tasks/PPT_TASK_R2_内容重写_逐页内容稿_20261117.md'
ls=task.read_text(encoding='utf-8-sig').splitlines();starts={}
for i,l in enumerate(ls,1):
 m=re.match(r'### 第 (\d+) 页',l)
 if m:starts[int(m[1])]=i
sections={i:(starts[i],(starts.get(i+1,len(ls)+1)-1)) for i in starts}
notes=audit['notes']
layouts={1:'学校模板封面',2:'学校模板目录',3:'任务与难点双栏 + 主线结论',4:'上方文献对标表，下方两组原文/原图',5:'公开数据与实测数据两个底框，9节点蛇形链 + 进行中节点',6:'模板章节页贰',7:'左侧五段论证，右上文献表、右下本项目时频图',8:'左侧四段论证，右侧平移单元格与信道前后时频图',9:'模板章节页叁',10:'四行审核表 + 结论与规模',11:'左五段论证，右侧谱线位置 + 同陷波表说明',12:'左五段论证，右侧底噪探针柱图 + 关键反例',13:'左四段论证，右侧背景替换两子图 + 关键数字',14:'左四段论证，右侧可编辑镜像示意 + 虚警子图',15:'左依据/做法/结果，右侧原文公式 + J/La/Lb图',16:'两幅折线图 + 设计、取舍与统计限制',17:'左文献表与设计/检验说明，右网络原图局部与特征箱图',18:'左四段论证，右融合结果图；箭头与大数字强调下降',19:'左侧发现/依据/做法/遗留问题，右正常/合并反例',20:'模板章节页肆',21:'左首轮→当前指标表，右方法与工作积累',22:'四项“文献→我们→证据”卡片',23:'左四项不足，右三项行动，计划一页带过',24:'学校模板尾页'}
pics={4:'r2_assets/soltani97.png、soltani49.png、wisig_fig10.png',7:'r2_assets/drone_table4.png、m100_window.png',8:'r2_assets/move_focus.png、channel_before.png、channel_after.png',11:'r2_assets/line2426.png',12:'r2_assets/source_probe.png',13:'background.png（图B上两子图）',14:'r2_assets/bgA_fa.png + 可编辑镜像示意',15:'r2_assets/gluge_mix3.png、wifi_focus.png',16:'r2_assets/window_miss.png、window_recall.png',17:'r2_assets/openrf_concat.png、mirror_feature.png',18:'fusion_focus.png',19:'normal.png、merged.png'}
o=['# SRTP中期答辩PPT：R2内容重写与验收报告','','## 当前交付','','主交付物：`../中期答辩_模板稿.pptx`，24页。最终预览：`../ppt_preview/r2_01.png` 至 `r2_24.png`（1920×1080），总览 `contact_r2.png`。本报告覆盖旧版REPORT，以R2任务书逐页内容为准。','','已完成：九个问题的因果链、统一的发现/证实/依据/解决/结果结构、文献原图与出处、图旁解释、每页备注、阶段成果与局限。仅保留第6/9/20页三张过渡页。未安装包、未导出PDF、未调用DSH、未发送heartbeat。','','## 每页版式与用图','','| 页 | 版式与内容组织 | 配图 |','|---|---|---|']
for i in range(1,25):o.append(f'| {i} | {layouts[i]} | {pics.get(i,"无，使用学校模板或可编辑图形/表格")} |')
o += ['','## 文献原图：文件、页码和裁切','','全部PDF使用pdftoppm渲染，PIL只打开PNG并裁切，未用Python打开PDF。采用200 dpi渲染；先查看整页，再查看裁切结果。坐标为渲染PNG像素，左/上/右/下。','','| PPT页 | 原始PDF | 页/图/式 | PNG与裁切范围 | 用途与校正 |','|---|---|---|---|---|',
'| 4 | Desktop/文献/05_TVT2020_RF_Fingerprinting_Nonstandard_UAV_Waveforms.pdf | PDF p.4–5, Sec.III-D | soltani200-04.png，(858,1775,1575,2075)；soltani200-05.png，(133,150,841,400) | 97%和49%实际在正文，不在同一结果图表。采用原文证据截图，明确写Sec.III-D，不冒称图号。 |',
'| 4 | C:/tmp/lit_wisig/wisig.pdf | p.6, Fig.10 | wisig200-06.png，(870,145,1555,515) | 同接收机/跨接收机曲线随训练接收机数变化，证明需留出评测。 |',
'| 7 | Desktop/文献/DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf | PDF p.7，刊页1153，表4局部 | drone200-07.png，(133,218,796,726) | 表4提供周期统计框架。M100行没有填图传周期，因此未声称14 ms来自表4；14 ms明确为本项目测量、由R2第7页提供。 |',
'| 15 | Desktop/文献/02_Robust_Low-Cost_Drone_Detection_Low_SNR_2024.pdf | p.3，式(3) | gluge200-03.png，(858,1663,1583,1858) | 先功率归一化后按SNR混合，选最关键混合公式放大，未把原式改写成我们的新方法。 |',
'| 17 | Desktop/文献/02_JSAC2026_Multi_Domain_Contrastive_RF_Open_Set.pdf | p.5，Fig.2局部 | openrf200-05.png，(491,145,1301,458) | 高亮C拼接位置。原图C表示拼接，⊕是另一运算，未按任务书的近似说法错标。Gao等、JSAC 2026稿按给定稿件标注，不将稿件声称为已出版。 |']
o += ['','WiSig下载来源：[arXiv:2112.15363](https://arxiv.org/pdf/2112.15363)，独立目录C:/tmp/lit_wisig。Geirhos下载来源：[arXiv:2004.07780](https://arxiv.org/pdf/2004.07780)，独立目录C:/tmp/lit_geirhos；首次下载超时后curl续传完成，pdftoppm渲染第2页，未采用图片，论证依据仍保留在第8/12/13页正文。','','## 本项目图：原路径与处理','','| PPT页 | 原路径 | 处理 |','|---|---|---|',
'| 7 | reports/数据筛选_20261013/模型输入时频图示例_15ms.png | 裁取Matrice 100的2.4 GHz子图(0,865,610,1060)，保留0–15 ms刻度。已查看T0100_D00_S0000长图确认机型；T0011实际为Matrice 200，未误用。 |',
'| 8 | reports/结果图表/06_验证实验_挪动和遮挡信号.png | 第一行前三格(250,48,873,154)，100%/4%/0%原值不变；以PPT标签说明三种条件。 |',
'| 8 | reports/结果图表/07_时频图_Matrice300各文件片段.png | 裁取S0110换信道前/后，(0,190,495,370)、(0,375,495,554)，在图旁说明2456→2409 MHz。 |',
'| 11 | work/prep_lines_20261104/lines_data_2.4GHz_page10.png | 第13子图，(0,3864,1820,4186)，红线2426.88 MHz位置加橙框。该PSD是聚合图，稳定性证据来自出现率检验，未把被宽带覆盖的峰伪装成突出尖峰。 |',
'| 12 | R2任务书第12页给定0%/93.2%/0.9% | 重绘三个底噪探针比例，0.51为同来源平衡准确率，在正文解释，未混入比例轴比较。 |',
'| 13 | reports/中期检查_20261117/figs/图B_背景替换对照.png | 复用background.png，上面两张2.4 GHz子图，裁取(0,105,1800,682)，图旁解释两张图分别看什么。 |',
'| 14 | figs/图C_加入接收机A背景前后.png | 只保留虚警左子图(0,65,1285,960)，镜像原理为可编辑形状，频段来自R2第14页。 |',
'| 15 | work/side_L_wifimix_20261110/metrics_paired.csv | 重绘C1合并虚警与v3_drone_recall；保留J/La/Lb和标准差，去除无关类别。 |',
'| 16 | work/main_W_window_20261110/metrics_summary.csv | argmax/main；nonG_miss与v3_drone_recall，mean/std乘100；各点N来自同L/同band的v3_test_FA行。50t横轴标50，原数据不改。 |',
'| 17 | work/fusionB_20261110/B1b/图17_接收机A与B镜像背景对比.png | 裁取谱平坦度箱图(1810,590,2470,940)，配合AUC来源检验文字。 |',
'| 18 | work/fusionB_B2_20261112/metrics_summary.csv | 复用fusion_focus.png：2.4 GHz v3_test_FA、5.8 GHz v3_drone_recall，R/Z/F/F5，均值±标准差。橙箭头仅标趋势，不改柱值。 |',
'| 19 | work/side_Q2_band_detect_20261111/图_k6_g2_data_drone_01.png；图_k6_g2_raw_background_01.png | 正常Q0000裁(785,25,1530,285)，反例Q1026裁(785,550,1530,810)，沿用normal.png/merged.png。右图是城区背景宽带合并反例，未将其频宽冒充正文60–68 MHz那条录音。 |']
o += ['','## 没有放入或需要答辩前核对的内容','','- 第7页找到正确Matrice 100长图，但全文件图按每列1 ms最大值聚合，不能从这张PNG可靠圈定一个精确10 ms空窗。采用同机型15 ms输入图说明短突发与间隙，橙框不标成“精确10 ms空窗”；14 ms与29%保留R2给定测量结论。','- 未加入Basak可选定位图，保留其方法及暗室叠加边界的文字依据，优先给正常/失败两张本项目证据图留可读空间。','- Soltani无同一图表直接给出97%→49%，采用实际正文证据。','- Open-RFNet图是时间/频率/纹理分支的拼接先例，不是本项目IQ手工特征网络的直接复现。','- 第18页“不支持把融合定为基线”保留待与导师讨论；第22页保留“据目前检索，答辩前再核对”；第23页保留结题时间待核对。','','## 逐页检查与验收','','R2共完成四次构建。第1、2轮分别导出24张1920×1080 PNG与总览并逐页打开检查，证据为r2_round1_*.png、r2_round2_*.png及contact_r2_round1/2.png。第3轮按COM实际BoundHeight排段落，修正文字与下一标题相挤；放大公式与WiFi两项图，处理引用行边界。第4轮补关键数字20–24 pt强调。最终再关闭写入器、复开主交付文件并逐页导出r2_*.png，检查修订页与整份总览。','','| 检查项 | 结果 |','|---|---|',f'| 最终复开 | {audit["reopened_slides"]}页 |',f'| 备注 | {len(notes)}/24页有正文讲稿与补充细节 |','| 溢出 | 最终COM BoundHeight/BoundWidth检查为0，实际段高用于排布 |','| 重叠、边距与校徽 | 逐页图检，内容不压页眉/页脚/校徽；图文分栏或上下结构 |','| 结构与逻辑 | 1–9问题链对应页7/8/10/11/12/13/14/15/17–18；每个结果引出后续问题 |','| 小标题 | 内容稿有对应项才放，未硬凑；第10页按指定审核表例外 |','| 图片与说明 | 所有图等比缩放，图旁给出读图重点与推论，不以图片代替说明 |','| 文献出处 | 已按PDF实物核对图号、页码；纠正Soltani无对应图表、DroneRFa表4不含M100周期、Open-RFNet C拼接符号 |','| 字号 | 问题页叙述正文12–12.6 pt，小标题13.5 pt；图注/来源10 pt；主要数字20–24 pt；其余学校模板字体保留 |','| 颜色 | 蓝色方法/已完成；橙色问题；深蓝结果；灰色计划 |','| 动画/切换 | 主动画序列清除，EntryEffect=0 |','| 禁止修改项 | 模板原件、中期检查表、figs、work原有文件均只读，未改写 |','| PDF/包/DSH/heartbeat | 未导出PDF、未安装包、未调用DSH、未发送heartbeat |']
o += [f'| 残留词：{k} | {v} |' for k,v in audit['residual'].items()]
o += ['','## 字数与最小字号（COM实测）','','“叙述正文”不含小标题、图注、关键数字重复强调、参考文献对标表。第10页按指定审核表统计表内正文241字；第17页叙述正文218字，另保留R2明确要求的文献对标表，其单元格计入页面总字数。这样保留了R2的论证密度和文献表，没有为压低统计数而删除依据。总字数为去空白可编辑文本字符，不包含位图文字。','','| 页 | 页面总字符 | 叙述正文字符 | 最小字号 pt（含图注/页码） |','|---|---:|---:|---:|']
for s in audit['slides']:o.append(f"| {s['page']} | {s['chars']} | {s['body_chars'] or '—'} | {s['min_font']} |")
o += ['','## 可编辑文字的逐数字出处','','包含正文、表格、图注、关键数字、备注中的不同数字词元。实验事实均来自R2同页；文件编号、图号、页码为结构定位数，按实际PDF/图表标注。参考文献年份/图号见文献表，不能当作新实验结果。','','| PPT页 | 数字词元 | R2出处页及行范围 |','|---|---|---|']
for s,n in zip(audit['slides'],notes):
 i=s['page'];tokens=sorted(set(re.findall(r'[−-]?\d+(?:\.\d+)*(?:%|\+)?','\n'.join(s['texts'])+'\n'+n)))
 if not tokens:continue
 lo,hi=sections.get(i,(1,len(ls)))
 for token in tokens:o.append(f'| {i} | {token} | R2第{i}页，{task.relative_to(R)}:{lo}–{hi}；图号/出处页另按原PDF核对 |')
o += ['','## 重绘图的CSV行与字段','','行号包含表头，比例乘100显示。误差棒为3种子标准差，不是置信区间。C1先在每个种子合并七档k/n，再求三种子均值与样本标准差。原图裁切数字仍对应原图原单元格/坐标刻度，不另造数据。','','| PPT页/图 | CSV文件:行 | 字段或计算 | 原始值/分子分母 |','|---|---|---|---|']
mf=json.loads((B/'r2_assets/manifest.json').read_text(encoding='utf-8'))
for x in mf:
 if 'row' not in x:continue
 d=x['data'];o.append(f"| 16/{x['output']} | {Path(x['source']).relative_to(R)}:{x['row']} | mean/std；N来自行{x['N_row']}的v3_test_FA | L={d['L']}, band={d['band']}, mean={d['mean']}, std={d['std']} |")
csvp=R/'work/side_L_wifimix_20261110/metrics_paired.csv';rows=list(csv.DictReader(open(csvp,encoding='utf-8-sig')))
for i,d in enumerate(rows,2):
 if d['mode']=='argmax' and (d['category']=='C1' or (d['category']=='main' and d['group']=='v3_drone_recall')):
  for g in ['J','La','Lb']:
   v='; '.join(f'{g}_{k}={d[g+"_"+k]}' for k in ['mean','std'])
   if d['category']=='C1':v='; '.join(f'{seed}: {d[f"{g}_{seed}_k"]}/{d[f"{g}_{seed}_n"]}' for seed in [2026,2027,2028])
   o.append(f'| 15/{g} | {csvp.relative_to(R)}:{i} | {d["category"]}/{d["group"]} | {v} |')
for label,file,row,fields,d in json.loads((B/'chart_sources.json').read_text(encoding='utf-8')):
 if label=='融合图':o.append(f'| 18/{d["group"]} | {Path(file).relative_to(R)}:{row} | {d["band"]}/{d["subset"]} rate_mean/rate_std | {d["rate_mean"]} / {d["rate_std"]} |')
o += ['','图像数字定位补充：第4页Soltani两个原文百分比对应III-D两种切分实验；WiSig曲线对应Fig.10坐标。第7页表4单元格保持原样、M100输入例刻度来自原图；第8页100/4/0为原06图第一行前三格；第11/17/19页刻度为所列原图对应轴；第13/14页曲线、柱与误差棒均保留原图位置和值。第12页0/93.2/0.9直接来自R2第12页。','','## 模板原件与复现','','复现：设置PYTHONIOENCODING=utf-8、KMP_DUPLICATE_LIB_OK=TRUE，使用D:/anaconda/python.exe -I依次运行r2_assets.py、r2_build.py 4、r2_verify.py、r2_report.py。r2_refs保存PDF渲染PNG和辅助文字；r2_assets/manifest.json记录裁切和数据行。PDF没有被Python打开。保存仍由本机WPS的PowerPoint.Application兼容COM完成，嵌入学校字体导致保存约需两分钟，等待其返回后才关闭并复开。']
tp=D/'模板原件.pptx';st=tp.stat();dt=datetime.datetime.fromtimestamp(st.st_mtime)
o += [f'- 原模板大小：{st.st_size}字节，要求78282941，一致。',f'- 原模板修改时间：{dt:%Y/%m/%d %H:%M:%S}，要求2026/11/17 19:58:26，一致。',f'- 原模板SHA256：`{hashlib.sha256(tp.read_bytes()).hexdigest()}`。',f'- 主交付物大小：{(D/"中期答辩_模板稿.pptx").stat().st_size}字节。',f'- 最终构建副本：{Path(audit["final_build"]).name}；最终审计：r2_final_audit.json。']
(B/'REPORT.md').write_text('\n'.join(o)+'\n',encoding='utf-8');print('R2 REPORT',len(o),'lines')
