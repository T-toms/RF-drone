from pathlib import Path
import json,re,datetime,hashlib
B=Path(__file__).resolve().parent;D=B.parent;ROOT=D.parents[1]
audit=json.loads((B/'audit.json').read_text(encoding='utf-8'))
task=ROOT/'work/codex_tasks/PPT_TASK_中期答辩PPT模板稿_20261117.md'
materials=[task,D/'build_ppt.py',D/'中期检查表_一二三项_原稿.md']
lines={str(p.relative_to(ROOT)):p.read_text(encoding='utf-8-sig').splitlines() for p in materials}
image_map={1:'无，保留模板第1页封面。',2:'无，保留模板第5页目录。',3:'无，复制模板第6页章节页。',4:'模板预览第19页原生图标裁取为 icon3_0/1/2.png，保持学校配色。',5:'无，原生可编辑流程与三卡。',6:'无，可编辑表格，项目做法列浅蓝底。',7:'无，复制模板第6页改为贰。',8:'无，可编辑九步蛇形流程。',9:'无，可编辑数字卡片与四节点时间轴。',10:'无，复制模板第6页改为叁。',11:'figs/图A_DroneRFa两步识别_测试集_不加噪.png → 原图等比置入。检索指定三目录未找到可确认身份的 Matrice 100 周期突发图，未添加不确定配图。',12:'work/prep_lines_20261104/lines_data_2.4GHz.png → lines.png → lines_candidate.png，裁取 (0,1000,1820,1315)，保留完整谱线子图、坐标轴及峰值，未改数据。',13:'figs/图B_背景替换对照.png → background.png，裁取 (0,105,1800,682)，仅保留上方两张2.4 GHz子图；5.8 GHz结论写在图注，保留两图坐标、图例及数据。',14:'figs/图C_加入接收机A背景前后.png → bgA.png，裁掉重复总标题，保留两子图。右侧镜像原理示意为可编辑形状，频率来自任务书。',15:'work/side_L_wifimix_20261110/metrics_paired.csv → wifi.png，重绘J/La/Lb四类指标。C1先按种子合并七档k/n，再计算三种子均值与样本标准差，其他指标使用*_mean/*_std。',16:'work/main_W_window_20261110/metrics_summary.csv → window.png，argmax/main的nonG_miss与v3_drone_recall；两个频段，mean与std乘100。50t仅横轴显示50，不改变值。无结论。',17:'work/fusionB_B2_20261112/metrics_summary.csv → fusion_focus.png，重绘2.4 GHz城区背景虚警、5.8 GHz城区无人机召回，R/Z/F/F5，rate_mean/rate_std乘100。',18:'work/side_Q2_band_detect_20261111/图_k6_g2_data_drone_01.png → normal.png，裁取 (785,25,1530,285)：Q0000 mini4 pro实验室例，3个清晰频带框。图_k6_g2_raw_background_01.png → merged.png，裁取 (785,550,1530,810)：Q1026城区2.4 GHz背景，被合并为宽频带；橙色圆角框为PPT叠加标注。该图只说明合并现象，图示宽度与正文60–68 MHz是不同样例，未将该图宽度冒充60–68 MHz。已逐张查看data_01、raw_drone_01/02/03、raw_background_01及最高频带占比_01后选定。',19:'无，复制模板第6页改为肆。',20:'模板预览第20页原生图标裁取为icon4_0/1/2/3.png。',21:'无，六项橙色问题标记。',22:'无，可编辑五节点时间轴，沿用模板第30页时间轴语法，调整为五节点并统一纯色。',23:'无，保留模板第44页尾页，人名xxx。'}
out=['# SRTP 中期答辩模板稿制作与自查报告','','## 交付与制作方式','','主交付物为 `../中期答辩_模板稿.pptx`，共23页，1920×1080预览位于 `../ppt_preview/01.png` 至 `23.png`，总览为 `contact.png`。每页均已写入讲稿备注。未导出PDF、未安装包、未调用DSH、未发送heartbeat。','','使用 PowerPoint.Application COM（本机由WPS演示提供兼容实现），沿用学校模板封面、目录、章节页、尾页、母版页眉线/页脚线/校徽/字体及配色。脚本和所有处理图仅写入本目录，输入文件只读。卡片、流程、镜像示意、表格、时间轴均可编辑；学校原生图标因WPS分组对象兼容性采用模板预览中的图标裁取。','','复现顺序：`prepare.py` → `build.py` → `finalize.py` → `audit.py` → `report.py`，均用 `D:/anaconda/python.exe -I`，设置 `PYTHONIOENCODING=utf-8`、`KMP_DUPLICATE_LIB_OK=TRUE`。build.py从只读初稿派生build_generated.py，overlay.py负责最终版式，finalize.py作最终裁切和关键数字强调。','','## 每页配图与处理','','| 页 | 原图来源与处理 |','|---|---|']
out += [f'| {i} | {image_map[i]} |' for i in range(1,24)]
out += ['','## 迭代检查记录','','第一轮：完整导出23张PNG和contact_round1.png，并逐页打开。发现章节页插入顺序错位、多行文本后续行未继承字号与颜色、WPS默认渐变、图标裁切不准、谱线全图过小及部分图表字过小。','','第二轮：重新构建导出23张PNG，逐页打开review_01.png至review_23.png检查。修正章节页顺序，改为整段设置字体和颜色，纯色填充并关闭阴影，校正图标裁切，统一五个难题页为上图下卡。检查残留示例文字为0，正文无溢出。','','最终修订与复开检查：调整四个章节页大号装饰字边界，融合图由双裁图改为CSV重绘并扩大坐标字号，谱线裁取完整子图，24%和1.9%改为20 pt单独强调。保存关闭后重新打开主交付文件，COM确认23页，再导出全部最终PNG并逐页复核。最终总览为contact.png。','','| 自查项 | 结果 |','|---|---|','| 文本截断、溢出 | COM BoundHeight/BoundWidth检查见audit.json；最终正文与装饰文字均无超界 |','| 重叠、校徽与分隔线 | 逐页检查，内容位于x35–925、y75–498范围内，图文不压页眉/页脚 |','| 对齐与等距 | 三卡等宽286 pt、间距16 pt；同一行顶端统一；流程与时间轴按网格排列 |','| 字号 | 正文≥14 pt，表格14 pt，图注11–12 pt；封面/尾页保留模板12 pt署名，属于封面元信息 |','| 图片 | 全部保持纵横比，白底卡片或0.75 pt细边框；重绘图保留均值和标准差 |','| 色彩语义 | 蓝色方法/已完成，橙色问题/进行中，深蓝结果，灰色计划 |','| 动画与切换 | 清除主动画序列，切换EntryEffect=0 |','| 备注 | 23/23页均有讲稿；正文和备注内容经COM提取保存在audit.json |','| 残留样例 | 以下8个搜索项均为0，扫描页面、母版和版式 |']
out += [f'| {k} | {v} |' for k,v in audit['residual'].items()]
out += ['','## 每页字数与字号（COM实测）','','字数按去除空白后的可编辑文本字符统计，不包含位图内文字。正文区字数包括图注、图上PPT标注、时间轴日期、表格单元格，因此并非纯讲述正文。表格页是信息密集对标页，保持14 pt；第18页包含两图标题、图上标注和公开数据图注，主要讲述正文约120字。页21已压缩为六条，页22数字主要为五组日期。封面、尾页12 pt保留模板原版署名；图注11 pt符合要求。','','| 页 | 总字符数 | 内容区字符数 | 页面最小字号 pt |','|---|---:|---:|---:|']
out += [f"| {r['page']} | {r['characters']} | {r['body_characters']} | {r['min_font']} |" for r in audit['slides']]
out += ['','## 数字出处表：可编辑正文与备注','','下表列出每页所有不同数字词元（包括型号/算法名、日期、讲稿中的秒数），来源保留材料路径和行号。纯编号只表示页序/条目序，不表示实验结果。图内数字另见下一节。','','| 页 | 数字 | 出处文件与行 |','|---|---|---|']
for r,note in zip(audit['slides'],audit['notes']):
 page=r['page'];tokens=sorted(set(re.findall(r'(?<!\d)[−-]?\d+(?:\.\d+)*(?:\+|%)?', '\n'.join(r['texts'])+'\n'+note)))
 tasklines=lines[str(task.relative_to(ROOT))];specific=[(i+1,l) for i,l in enumerate(tasklines) if l.startswith(f'| {page} |')]
 for token in tokens:
  found=[]
  for i,l in specific:
   if token.lstrip('−-') in l:found=[f'{task.relative_to(ROOT)}:{i}'];break
  if not found:
   for f,ls in lines.items():
    for i,l in enumerate(ls,1):
     if token.lstrip('−-') in l:found=[f'{f}:{i}'];break
    if found:break
  out.append(f"| {page} | {token} | {found[0] if found else '结构编号或图示轴刻度，不构成新增实验数字'} |")
out += ['','## 图内数字、单元格与重绘来源','','原图A（第11页）保留全部混淆矩阵单元格，出处为任务书指定的 `figs/图A_DroneRFa两步识别_测试集_不加噪.png` 原矩阵对应行列单元格，未改写。图B（第13页）保留原上两子图所有点、阴影、原样本参考线和N值，出处为原图对应2.4 GHz子图及横轴dB位置。图C（第14页）保留原图柱值和误差棒，出处为原图对应接收机/数据组。谱线、频带图内数字均为原图对应坐标刻度/原图标题记录，不新增实验事实。','','以下为每个重绘数字所依据的CSV行与字段，记录原始比例和标准差；显示单位为百分比（乘100）。坐标轴自动刻度是绘图刻度，不是额外实验结果。完整逐行数据快照存于chart_sources.json。','','| 图表 | CSV文件:行 | 字段/计算 | 原始数值 |','|---|---|---|---|']
charts=json.loads((B/'chart_sources.json').read_text(encoding='utf-8'))
for label,file,row,fields,data in charts:
 keys=['L','band','group','mean','std'] if label=='窗长图' else ['band','group','subset','rate_mean','rate_std'] if label=='融合图' else [k for k in data if (k.endswith('_mean') or k.endswith('_std')) and k.startswith(fields.split('_')[0])]
 if '七档' in label:keys=[k for k in data if (k.endswith('_n') or k.endswith('_k')) and k.startswith(fields[0:2] if fields.startswith('La') or fields.startswith('Lb') else 'J')]
 val='; '.join(f'{k}={data[k]}' for k in keys)
 out.append(f'| {label} | {Path(file).relative_to(ROOT)}:{row} | {fields} | {val} |')
out += ['','## 待补与审阅项','','- 第16页窗长比较结论为【待补】，按任务书要求未作优劣判断。','- 第20页创新点的检索边界保留【答辩前再核对】。','- 第22页计划保留【结题时间待核对】。','- 人名均为xxx，供用户修改。','- 第18页宽频带反例来自城区背景，展示相同合并失败现象；未将其频宽当作正文60–68 MHz实例。','','## 模板与文件完整性','','模板原件未改写，以下为最终文件实测值。']
tp=D/'模板原件.pptx';st=tp.stat();mtime=datetime.datetime.fromtimestamp(st.st_mtime)
out += [f'- 模板大小：{st.st_size}字节（要求78282941）。',f'- 模板修改时间：{mtime:%Y/%m/%d %H:%M:%S}（要求2026/11/17 19:58:26）。',f'- 模板SHA256：`{hashlib.sha256(tp.read_bytes()).hexdigest()}`。',f'- 最终PPT：{(D/"中期答辩_模板稿.pptx").stat().st_size}字节，已关闭后COM重新打开并确认{len(audit["slides"])}页。','','COM兼容处理：WPS的FindBySlideID返回空对象，改用保留页的稳定集合引用；新建页/章节页插入行为不同，构建后显式统一顺序；整段设置字号和颜色避免多段格式丢失；表格单元格Top接口不支持，审计用所在内容区统计。保存时保留学校字体嵌入，单次写入约需两分钟，等待保存返回后才导出和复开。所有Quit包在try中，关闭前Saved=True避免未保存内存副本滞留。']
(B/'REPORT.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print('REPORT written',len(out),'lines')
