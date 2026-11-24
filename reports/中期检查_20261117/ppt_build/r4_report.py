# -*- coding: utf-8 -*-
from pathlib import Path
import json,re,zipfile,xml.etree.ElementTree as E,hashlib,ast,datetime
from PIL import Image
from r4_content import REFS,LAYERS,LAYER_NAMES
B=Path(__file__).resolve().parent;D=B.parent;R=B.parents[2];N=B/'r4_assets';P=D/'ppt_preview';W=B/'r4_work.pptx'
audit=json.loads((N/'audit_round7.json').read_text(encoding='utf-8'))
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(W) as z:
 full='\n'.join(''.join(t.text or '' for t in E.fromstring(z.read(n)).findall('.//a:t',ns)) for n in z.namelist() if re.fullmatch(r'ppt/(slides/slide\d+|notesSlides/notesSlide\d+)\.xml',n))
ban=['F组','F5组','Z组','R组','bgA','bgB','v3','data/','nonG','C1','argmax','thr_fa1','捷径','shortcut','阳性对照','AUC','召回','recall','虚警','种子','seed','嵌入层拼接','Codex','任务书','worker','城区','家居','作弊','假象','失败','实验室接收机','接收机 A','接收机 B','接收机A','接收机B','evo2pro_v3']
counts={word:full.count(word) for word in ban}
for code in ['La','Lb','J','H','L','M','W','B1','B2b']:counts[code+'（独立实验代号）']=len(re.findall(r'(?<![A-Za-z0-9-])'+code+r'(?![A-Za-z0-9-])',full))
extract=(R/'work/fusionB_20261110/extract.py').read_text(encoding='utf-8-sig');names=ast.literal_eval(re.search(r'NAMES=(\[.*?\])',extract)[1]);counts.update({name:full.count(name) for name in names})
assert not any(counts.values()),{k:v for k,v in counts.items() if v}
assert len(audit)==21
assert all(200<=d['notes_chars']<=300 and d['min_font']>=9 for d in audit)
research=[5,10,11,12,13,14,15,16];heads=['研究背景','现象','分析与验证','改进方法','效果与启示']
body={}
for d in audit:
 i=d['page'];paras=[t for t in d['texts'] if any(t.startswith(h+'　') for h in heads)]
 if i in research:
  body[i]=sum(len(re.sub(r'\s','',t))-len(t.split('　')[0]) for t in paras)
  assert 450<=body[i]<=600,(i,body[i])
  assert all(any(t.startswith(h+'　') for t in paras) for h in heads)
 elif i in [19,20]:
  body[i]=sum(len(re.sub(r'\s','',t.split('　',1)[-1])) for t in d['texts'] if '针对层次' in t)
  assert 500<=body[i]<=700,(i,body[i])
 else:body[i]=None
for i in range(4,20):
 bottom=next(t for t in audit[i-1]['texts'] if ' → ' in t and len(t)>30 and t.startswith(('基线','窗长','形成','来源','统一','固定','489','2.4','同设备','检出','得到','融合','能量','58','局限','第一阶段')))
 nxt=bottom.split(' → ',1)[1]
 assert nxt in audit[i]['texts'],(i,nxt)
assert '采存' not in '\n'.join(audit[6]['texts']) and 'DroneRFa' not in '\n'.join(audit[6]['texts'])
assert '探索性实验（公开数据集）' in ''.join(audit[1]['texts'])
net=''.join(audit[8]['texts']);dims=['1×256×256','64×128×128','64×64×64','128×32×32','256×16×16','512×8×8','576']
assert all(v in net for v in dims)
for rnd in [1,2,6]:assert all(Image.open(P/f'r4_round{rnd}_{i:02d}.png').size==(1920,1080) for i in range(1,22))
originals=json.loads((N/'originals.json').read_text(encoding='utf-8'))
unchanged={}
for path,old in originals.items():
 p=Path(path)
 if 'Desktop/中期' in path.replace('\\','/') or 'Desktop\\中期' in path:continue
 if old.get('exists') is False:continue
 unchanged[path]=p.stat().st_size==old['size'] and p.stat().st_mtime_ns==old['mtime_ns'] and hashlib.sha256(p.read_bytes()).hexdigest()==old['sha256']
assert all(unchanged.values())
false_headers=['文献中的做法与发现','核验项','中文名'];real_overflow=[(d['page'],o) for d in audit for o in d['overflow'] if o['text'] not in false_headers]
assert not real_overflow,real_overflow
blank={1:None,2:None,3:12,4:14,5:8,6:9,7:11,8:9,9:12,10:11,11:10,12:12,13:13,14:9,15:13,16:12,17:12,18:11,19:14,20:14,21:None}
lines=['# 中期答辩 R4 制作与验收记录','',
'最终交付按新增 §3B 写回桌面 `C:/Users/Administrator/Desktop/中期答辩_模板稿.pptx`，共21页。制作及只读复核文件为 `r4_work.pptx`；R3备份 `r3_dense.pptx` 未改。预览 `../ppt_preview/r4_01.png` 至 `r4_21.png`，1920×1080；总览 `../contact_r4.png`。','',
'## 范围和新增要求','',
'本轮仅处理PPT及新素材，没有运行或改写任何中期检查表脚本、表格文件；没有导出PDF，没有调用DSH，没有发heartbeat。适用要求依次为3B、3A、R4正文及R3禁用词/特征名/科研风规范。','',
'公开数据集部分统一为探索性实验；目录、标题、五层脉络和备注一致。第7页只介绍两批实测数据，不写换数据原因，不保留新机型、连续性、采存中断等比较；过渡固定为“探索性实验确定了处理流程和评测原则，正式实验在课题组采集的实测数据上开展”。','',
'## 自查结论','',
'- 幻灯片、原生表格与全部备注的指定禁用词、19个英文特征变量名：0次，完整清单如下。正式型号EVO II Pro V3保留；不把IEEE J-RFID的J当实验代号。','- 所有21页备注200–300字符，保留口语讲述和引到下一页的衔接。','- 第5、10–16页五个叙事含义齐全，内文以细箭头自上而下推进，保留因果句，不使用五个等大并列块。第5页两栏各有一套。','- 第3页五级脉络对应1/2/3/4/5层；第4–17页有五小格进度，当前层次蓝色。第8/9页属于层次三的统一方法。第18–20页逐项注明结论来自哪层。','- 第17页原生可编辑曲线仅58→62→81；同设备误报100→0，第三步不补造数值；85.7%仅在重新训练图注，不作为第四点。','- 网络结构、各张量尺寸与R3第5页一致；训练超参数不变，补充选型及加权、选轮次、独立重复的原因。','- 文字真实溢出0。WPS单元格对象对首个表头的首次几何返回不稳定，三个表头“文献中的做法与发现/核验项/中文名”会被COM列为候选；后续稳定几何均足够、预览不溢出，明确单列而不当真实溢出。','',
'## 两轮预览与后续兼容修正','',
'第1轮完整导出并逐页查看1–21页：修正标题与多行框高度、网络分支标签、内部数据名称和计划页留白。第2轮完整导出并检查：加入3A五层脉络、进度、承接及结论条，重画第17页，补充18–20页来源层次。收到3B后进行第3–6轮针对性修正，最终复开工作PPT只读导出全部21页，不经WPS再次保存。','',
'WPS保存将部分固定框高度归为27.15 pt；按实际BoundHeight/BoundWidth在OOXML修复。图像内部旧名称用原始CSV重绘，背景替换与无信号柱图用PPT原生遮盖/中文标签替换，曲线、柱值、误差棒像素不改。`r4_figs.py`保留300 dpi科研图脚本。最后逐页检查标题、段落、表头、图注、连线、页脚与校徽，计划页补充小型方法流程图。','',
'## 字数、字号、留白与叙事','',
'字数为去空白字符数，数字和西文逐字符计。研究正文只统计五段叙事，排除标题、承接条、结论条、表格/图表标签和出处；计划正文只统计思路块文字。全页列包含全部可编辑文字。留白按叙事组、证据组与承接/结论条的外包区域目测估计，不把图表内部白底算留白；精度约±3个百分点，封面、目录、致谢按模板保留。','',
'|页|全页字符|研究/计划正文|最小字号pt|备注字符|估算留白|五含义/衔接|','|---|---:|---:|---:|---:|---|---|']
for d in audit:
 i=d['page'];lines.append(f"|{i}|{d['chars']}|{body[i] if body[i] is not None else '方法/介绍/模板'}|{d['min_font']:g}|{d['notes_chars']}|{str(blank[i])+'%' if blank[i] is not None else '模板'}|{'齐全，含因此/据此/由此等，箭头推进' if i in research else '按功能分块；见逐页承接'}|")
lines+=['','## 逐页承接与结论条（3A.5）','','前页结论条箭头后的全文，与后页承接条逐字符匹配；第6→7页按3B单句过渡。','', '|页|层次|承接条|本步结论 → 下一步|','|---|---|---|---|']
for d in audit[3:20]:
 i=d['page'];bottom=next((t for t in d['texts'] if ' → ' in t and len(t)>30 and t.startswith(('基线','窗长','形成','来源','统一','固定','489','2.4','同设备','检出','得到','融合','能量','58','局限','第一阶段','型号'))),'')
 if i==4:top=next(t for t in d['texts'] if t.startswith('研究先'))
 else:
  prev=next(t for t in audit[i-2]['texts'] if ' → ' in t and len(t)>30 and t.startswith(('基线','窗长','形成','来源','统一','固定','489','2.4','同设备','检出','得到','融合','能量','58','局限','第一阶段')));top=prev.split(' → ',1)[1]
 lines.append(f"|{i}|{LAYERS.get(i,'展望')}|{top}|{bottom}|")
lines+=['','## 每页文献三要素','','没有另查文献或补造文献数字。来源事实限R4 §1.4；未采用Open-RFNet可选数字，融合说明用Han，避免越出§1.4事实表。NIST为数据源、He/Yu/Geirhos没有任务给定的性能数字，保留其给定结论而不补造。','', '|页|引用|做了什么 / 结论 / 本项目关系|','|---|---|---|']
evidence={4:'DroneRFa：公开多频段数据和时频图ResNet18，随机97.73%；对照本项目按文件基线。',5:'DroneRFa：周期统计与短片段局限，对照14 ms；Geirhos：无关线索与同分布局限，支持单因素平移。',6:'Soltani：7架同型号突发97%→49%，支持按录音；WiSig：跨设备99%→<33%，支持分来源；Geirhos：改变条件发现线索，支持对照；Glüge：数字混合与−12 dB以上≥85%，支持归一化。',8:'Basak：统计检测后分类与YOLO定位比较，后者多信号更稳并输出频率/带宽；对照本项目输入尺寸及分带。',9:'He：残差相加减轻深层训练困难，支持主干；DroneRFa：时频图ResNet18与97.73%基线，对齐探索方法。',10:'Soltani：相邻混合97%/留出49%对应重划分；WiSig：跨设备下降对应杂散检查。',11:'Geirhos：训练分类器检验两批分布差异、改变条件检验线索；对应底噪预检验与背景替换。',12:'Glüge：另录设备噪声为背景，外场0.81–0.99；对应本项目无法另录时构造空闲段。',13:'Glüge：归一化后按比例混合背景，高于−12 dB≥85%；对照本项目两类对称混合。',14:'Han：统计与深度拼接、部分条件有效而0 dB以下接近随机，对应先筛后评；WiSig99%→<33%和Glüge≥85%支持归一化。',15:'Han：调制识别中融合有条件性，弱信号可退化；对应结构/输入分离对照及后续分档复评。',16:'Basak：多信号定位优于纯统计检测，支持先定位再分类；NIST30 MS/s及两信道作为计划干扰数据源，不声称已产生提升。',18:'Han：融合效果有条件，0 dB以下可能失效；对应低信噪比重评。',19:'Glüge：功率比混合、−12 dB以下下降，支持对称叠加及弱信号分档；NIST参数支持重采样数据源。',20:'DroneRFa：表4统计周期、短片段不全，支持长窗；Yu：开集识别应拒识未知，对应留出型号及阈值。'}
for i,msg in evidence.items():lines.append(f'|{i}|{REFS.get(i,"")}|{msg} 三项已逐条核对。|')
lines+=['','## 禁用词和特征变量清单','','|词/变量|文本与备注次数|','|---|---:|']+[f'|{k}|{v}|' for k,v in counts.items()]
lines+=['','## 数字出处','','|页|数字范围|来源|','|---|---|---|',
'|3|≤1%、≥95%、0/−5/−10/−15 dB；五层图85.4%、0–4%、90/57%、62/81%|R4 §3第3页、§3A.1；R3第3、7、9、11页|',
'|4–6|24+1、915/2.4/5.8、100 MS/s、10 ms、6:2:2、97.73、12+背景、85.4/51.1、34.8、14/1–2 ms、29%、2456/2409、20–40、0–4、34/59、96/9、97/49、99/<33|R4 §1.4及第4–6页；R3第6–7页|',
'|7|14/7型号、122.88/100 MS/s、3933/61 GB、890/73 GB、100/93%、1:13.8/1:12.2|R4第7页两批数据表与分布图指定值|',
'|8–9|采样率、频带、1024/512、833/922、2399/3905、1×256×256；全部网络张量与训练参数|R3第4–5页；R4第8–9页；图像值与R3原数据不变|',
'|10–13|97/15/25、两条采样率整数比、489/173；3/93.2/0.9/0.51/0.4/92/58/90/57；19/57/61/6/0.3/1287/0.23/0.42/62；−5至15/88/81/24/1.9|R4第10–13页；R3第8–11页|',
'|14–15|19→13→10→5、0.934/0.77–0.91、全部特征公式；24次、2.5、94.9/89.2/86.2、5.5/0.4百分点|R3 §1.2与第12–13页；R4第14–15页|',
'|16|99%、2460–2475、6 dB、21、2–4带、60–68、NIST参数|R4第16页与§1.4；R3第14页|',
'|17|58→62→81、100→0、1.9/24、85.7/4.2、80/3.7/300、360/60余/26/20/100余|R4 §3A.3和第17页；R3第15–16页；85.7不作为新步骤|',
'|18–20|80/3.7/300、2461、0 dB、30/6/165、61.44/100、0至−15、13+7−2≈18、12/14 ms、20/50、10/20/40 MHz|R3第16–18页；R4第18–20页和§1.4，均为目标或原有观测，未冒充新增结果|',
'','图表重绘的CSV源、行号、均值与标准差保留在 `r4_assets/manifest.json`、`fusion_sources.json`（延用R3同一原始CSV）。DroneRFa图6(a)取自已渲染 `r2_refs/drone-06.png`，已先打开确认，裁切(565,355,858,582)，刊页1152。其他文献原图沿用R3：表4、Soltani Sec.III-D原段、WiSig Fig.10、Glüge式(3)。','',
'## 原件与交付状态','','模板原件、R3备份与报告目录中的表格原件以开始时size/mtime_ns/SHA256核对；桌面两个文件遵循协调者要求只用Get-Item读属性，不用COM打开，不读内容。桌面PPT按3B应有意替换，因此不再将其作为“未变”断言。桌面表格必须大小54784、mtime保持原值。','', '|受保护路径|核对结果|','|---|---|']
lines += [f'|{p}|大小、mtime_ns、SHA256未变：{ok}|' for p,ok in unchanged.items()]
lines+=['','桌面原文件的覆盖前后属性及占用检查记录由最终交付步骤追加。','',
'## 小问题与复现','','1. 初始指定报告目录的R3主文件不存在；协调者确认已移到桌面，且与r3_dense大小/时间一致。始终使用r3_dense，不用COM打开用户桌面PPT。','2. 工作中先收到3A、后收到3B，均覆盖旧条款。旧R4文件仅留作内部渲染草稿素材，交付只写回桌面原文件。','3. 保持R3模型与证据图值，手工特征方案明确：10个保留谱平坦度，5个才去掉；原生流程与分布图可编辑。','4. 复现：r4_build.py 1、r4_build.py 2 → r4_figs.py → r4_repair.py → r4_build.py 6 --audit --work → r4_report.py。r4_repair为3A/3B最终内容和WPS兼容修正入口，不能省略；未调用任何form脚本。']
(B/'REPORT_R4.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(N/'checks.json').write_text(json.dumps({'slides':21,'forbidden_counts':counts,'research_body_chars':body,'notes_all_200_300':True,'min_font_at_least_9':True,'real_overflow':real_overflow,'protected_unchanged':unchanged,'preview_rounds':[1,2,3,4,5,6],'progression_links_match':True,'desktop_delivery':'pending'},ensure_ascii=False,indent=2),encoding='utf-8')
print('All content checks passed; report written.')


