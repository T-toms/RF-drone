# -*- coding: utf-8 -*-
"""R4 template-preserving WPS COM authoring, two-round PNG review, no PDF."""
from pathlib import Path
import re,json,hashlib,sys,math
import win32com.client as win32
from PIL import Image,ImageDraw
from r4_content import C,TITLES,REFS,LEADS,BRIDGES,LAYERS,LAYER_NAMES,TAKEIN,STEP_RESULT
B=Path(__file__).resolve().parent;D=B.parent;ROOT=B.parents[2];P=D/'ppt_preview';A=B/'r3_assets';NEW=B/'r4_assets'
NEW.mkdir(exist_ok=True)
OUT=D/'中期答辩_R4.pptx'
if '--work' in sys.argv:OUT=B/'r4_work.pptx'
SOURCE=D/'中期答辩_模板稿.pptx'
if not SOURCE.exists():SOURCE=B/'r3_dense.pptx'
ROUND=int(sys.argv[1]) if len(sys.argv)>1 else 1
def snapshot():
 paths=[D/'模板原件.pptx',D/'中期答辩_模板稿.pptx',SOURCE,D/'中期检查表_原件.doc']
 return {str(p):({'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} if p.exists() else {'exists':False}) for p in paths}
def rgb(r,g,b):return r+256*g+65536*b
BLUE=rgb(0,121,186);PALE=rgb(232,242,250);GRAY=rgb(120,120,120);TXT=rgb(64,64,64);WHITE=rgb(255,255,255);ORANGE=rgb(240,163,10)
def style(tr,size=12,c=TXT,b=False):
 tr.Font.NameFarEast='微软雅黑';tr.Font.Name='Arial';tr.Font.Size=size;tr.Font.Color.RGB=c;tr.Font.Bold=-1 if b else 0
def tx(s,x,y,w,h,text,size=12,c=TXT,b=False):
 sh=s.Shapes.AddTextbox(1,x,y,w,h);sh.Name='r4_text';tf=sh.TextFrame;tf.MarginLeft=tf.MarginRight=1;tf.MarginTop=tf.MarginBottom=1;tf.WordWrap=-1;tf.AutoSize=0;tf.TextRange.Text=text;style(tf.TextRange,size,c,b);tf.TextRange.ParagraphFormat.SpaceAfter=0
 return sh
def rect(s,x,y,w,h,fill=WHITE,dash=False):
 sh=s.Shapes.AddShape(1,x,y,w,h);sh.Name='r4_diagram';sh.Fill.Solid();sh.Fill.ForeColor.RGB=fill;sh.Line.ForeColor.RGB=TXT;sh.Line.Weight=.8;sh.Shadow.Visible=0
 if dash:sh.Line.DashStyle=4
 return sh
def line(s,x,y,xx,yy,arrow=True):
 sh=s.Shapes.AddLine(x,y,xx,yy);sh.Line.ForeColor.RGB=TXT;sh.Line.Weight=.8
 if arrow:sh.Line.EndArrowheadStyle=3
def pic(s,path,x,y,w,h,caption=''):
 path=Path(path);path=path if path.is_absolute() else A/path
 im=Image.open(path);ratio=min(w/im.width,h/im.height);ww=im.width*ratio;hh=im.height*ratio
 s.Shapes.AddPicture(str(path),0,-1,x+(w-ww)/2,y+(h-hh)/2,ww,hh)
 if caption:tx(s,x,y+h+3,w,40,caption,10,GRAY)
def table(s,rows,x,y,w,h,widths,size=11):
 sh=s.Shapes.AddTable(len(rows),len(rows[0]),x,y,w,h);sh.Name='r4_table';tb=sh.Table
 for j,ww in enumerate(widths,1):tb.Columns(j).Width=ww
 for i,row in enumerate(rows,1):
  for j,t in enumerate(row,1):
   cell=tb.Cell(i,j).Shape;tf=cell.TextFrame;tf.TextRange.Text=t;style(tf.TextRange,size,WHITE if i==1 else TXT,i==1);tf.MarginLeft=tf.MarginRight=4;tf.MarginTop=tf.MarginBottom=3;tf.WordWrap=-1;tf.TextRange.ParagraphFormat.SpaceAfter=0;cell.Fill.Solid();cell.Fill.ForeColor.RGB=BLUE if i==1 else WHITE
 return sh
def prose(s,items,x=35,y=99,w=540,h=384,size=12):
 shapes=[]
 for head,body in items:
  sh=tx(s,x,y,w,500,head+'　'+body,size);tr=sh.TextFrame.TextRange;style(tr.Characters(1,len(head)),size,BLUE,True);shapes.append(sh)
 heights=[sh.TextFrame.TextRange.BoundHeight+3 for sh in shapes];needed=sum(heights)
 if needed>h:
  raise RuntimeError(f'prose overflow page {s.SlideIndex}: {needed}>{h}')
 gap=min(15,(h-needed)/max(1,len(shapes)-1));yy=y
 for k,(sh,hh) in enumerate(zip(shapes,heights)):
  sh.Top=yy;sh.Height=hh
  if ROUND>=2 and k<len(shapes)-1 and len(items)>=3:line(s,x-8,yy+hh-1,x-8,yy+hh+gap-1)
  yy+=hh+gap
 return shapes
def node(s,x,y,w,h,text):
 rect(s,x,y,w,h,PALE);sh=tx(s,x+5,y+6,w-10,h-8,text,11);return sh
def generic(s,i):
 prose(s,C[i]);
def notes(i):
 if i in LEADS:
  return LEADS[i]+'本阶段先在公开数据集建立时频图和残差网络基线，再将周期分析、按录音划分与单因素对照延续到实测数据。我们既报告检出和误报，也检查判断是否受环境、频率位置或设备噪声影响。当前同设备背景构造与对称干扰叠加已有实验依据，但手工特征融合尚未稳定提升，背景数量也不足以证明低于百分之一的误报目标。后续将扩充背景、完善信号频带截取，分信噪比评估，再开展型号识别、跨条件测试和未知型号拒识。希望各位老师重点指导评测边界与后续方案。'
 items=C.get(i,[])
 if i==9:
  return '这一页介绍网络结构和训练设置。时频图分支沿用公开数据集的ResNet18，通过残差连接减轻深层网络训练困难。手工特征由独立分支投影为64维，与512维图像特征拼接后分类；只用时频图时直接连接分类头。由于背景较少，训练采用类别加权，并按验证集误报与漏检之和选择轮次，每组独立训练三次，检查改善方向是否一致。需要强调，增加分支本身也会改变头部结构，因此后面的融合评估会保留特征全置零对照，区分结构变化与特征信息的作用。接下来先回到数据端，说明质量核验和设备杂散处理。'
 lead='这一页说明'+TITLES[i]+'。'
 # Pick whole factual sentences, with evidence before the closing transition.
 sentences=[]
 for idx in ([0,1,2,3,4,5] if len(items)>4 else range(len(items))):
  if idx>=len(items):continue
  ss=re.split('(?<=[。；])',items[idx][1]);sentences.extend(ss[:2])
 text=lead
 for sent in sentences:
  if len(text+sent)+len(BRIDGES.get(i,''))<=285:text+=sent
  if len(text)>220:break
 text+=BRIDGES.get(i,'')
 if len(text)<200:text+='这里的判断都保留按录音划分、分来源统计和独立重复的要求，防止总体指标掩盖不同条件下的差异。'
 return text

def progression(s,i):
 if i<4 or i>20:return
 # Reflow the body to reserve two explicit transition bands.
 for sh in list(s.Shapes):
  yy=float(sh.Top)
  if yy<95 or yy>=496:continue
  sh.Top=129+(yy-100)*.87
  oldh=float(sh.Height);sh.Height=oldh*.87
  if sh.HasTable:
   for cell in walk([sh]):
    if cell.HasTextFrame:
     tr=cell.TextFrame.TextRange;tr.Font.Size=max(9,float(tr.Font.Size)-.5)
  elif sh.HasTextFrame and sh.TextFrame.HasText:
   tr=sh.TextFrame.TextRange
   for k in range(1,tr.Runs().Count+1):
    run=tr.Runs(k);old=float(run.Font.Size)
    run.Font.Size=max(11 if old>=11 else 9,old-.5)
   if tr.BoundHeight>sh.Height:sh.Height=tr.BoundHeight+2
 # Flat bands and small progress blocks are explicitly requested navigation.
 band=rect(s,35,96,890,27,PALE);band.Line.Visible=0
 tx(s,43,102,874,19,TAKEIN[i],10.5,BLUE)
 layer=LAYERS.get(i)
 if layer:
  for k in range(1,6):
   box=rect(s,640+(k-1)*15,74,11,11,BLUE if k==layer else rgb(224,228,232));box.Line.Visible=0
  tx(s,720,71,205,19,'层次'+['零','一','二','三','四','五'][layer]+'：'+LAYER_NAMES[layer],10,BLUE,True)
 else:tx(s,640,71,285,19,'展望：走向弱信号与型号识别',10,BLUE,True)
 sh=rect(s,35,478,890,24,BLUE);sh.Line.Visible=0
 next_text=TAKEIN.get(i+1,'完成两阶段串联与跨条件评测，接受指导')
 # The exact next-page connector is repeated for an auditable narrative chain.
 tx(s,42,483,876,18,STEP_RESULT[i]+' → '+next_text,9.5,WHITE)

def build():
 before=snapshot();app=win32.DispatchEx('PowerPoint.Application');app.DisplayAlerts=1;p=app.Presentations.Open(str(SOURCE),False,False,False)
 try:
  layout=p.Slides(3).CustomLayout
  p.Slides.AddSlide(4,layout);p.Slides.AddSlide(7,layout)
  # Move the R3 input/network pages behind the two new public-dataset pages.
  # Current order: 1,2,3,new,old4,old5,new,old6,old7,...
  p.Slides(9).MoveTo(5);p.Slides(9).MoveTo(6);p.Slides(9).MoveTo(7)
  # now new data intro is page 7, old4 and old5 are page8 and page9.
  for i in range(3,21):
   s=p.Slides(i)
   if i!=9:
    for sh in list(s.Shapes):sh.Delete()
   else:
    for sh in list(s.Shapes):
     if sh.Top<75 or sh.Top>505:sh.Delete();continue
     # preserve editable network diagram and tensor dimension strings.
     sh.Top=157+(sh.Top-87)*.73
     if sh.Type not in [9]:sh.Height*=.73
   title=tx(s,35,16,748,42,TITLES[i],23,TXT,True);title.Name='r4_title'
   stage='公开数据集阶段' if i<=6 else ('实测数据阶段 · 环节 '+str(i-9) if 10<=i<=16 else '研究目标与方法' if i in [3,8,9] else '数据介绍' if i==7 else '成果、局限与计划')
   tx(s,35,70,890,20,stage,10,BLUE)
   if i in REFS:tx(s,35,497,855,19,REFS[i],9,GRAY)
   tx(s,900,516,25,15,str(i),10,GRAY)
   print('building page',i,flush=True)
   if i==3:
    prose(s,C[i][:2],35,103,390,222,12)
    for y,t in [(108,'公开数据集阶段\r基线、周期与频率位置\r确定评测原则'),(233,'实测数据阶段：第一阶段检测\r核验与预处理 → 时频图＋手工特征\r网络分类 → 判断依据检验 → 针对性改进'),(375,'第二阶段识别（计划）\r信号频带与中心频率、跨条件与未知型号')]:
     rect(s,453,y,472,87,WHITE,y==375);tx(s,466,y+12,445,67,t,12,BLUE)
    line(s,680,197,680,226);tx(s,453,202,472,22,'处理流程、评测原则延续到实测数据',10,GRAY)
    line(s,680,322,680,366);prose(s,C[i][2:],35,330,390,164,11.5)
   elif i==4:
    generic(s,i)
    im=Image.open(B/'r2_refs/drone-06.png');im.crop((565,355,858,582)).save(NEW/'dronerfa_fig6a.png')
    pic(s,NEW/'dronerfa_fig6a.png',600,104,325,225,'DroneRFa 图6(a)，刊页1152：Air 2S图传时频图，展示周期性发射。')
    tx(s,600,390,325,40,'85.4% / 51.1%',23,BLUE,True);tx(s,600,437,325,49,'本项目按文件划分\r不加噪 / −15 dB',12)
   elif i==5:
    # Two parallel narratives, five headings each, paired evidence below.
    left=[('研究背景','基线出现周期性漏检，先核对完整录音中的发射规律。'),('现象','Matrice 100每14 ms发1–2 ms突发，10 ms窗约29%为空。'),('分析与验证','DroneRFa表4统计周期，原文指出短片段观测不全；本项目据此逐机型核对，14 ms为本项目测量。'),('改进方法','先画整条长时图、剔除不含信号规律的窗口；15、20 ms窗不再出现空窗。'),('效果与启示','窗长应覆盖周期，延续到实测窗长比较及型号识别。')]
    right=[('研究背景','Matrice 300准确率34.8%，需检查信道变化。'),('现象','图传约2456→2409 MHz；位置变化与低检出对应。'),('分析与验证','Geirhos综述牛与草地等无关线索，指出同分布高分难以揭示影响，需改变条件。本项目不重训只平移20–40 MHz，形状不变，100%降至0–4%；挪回后34%升至59%，遮掉2400–2425 MHz使Mavic 3从96%降至9%。'),('改进方法','因此用频率平移与遮挡，每次只改变一个因素。'),('效果与启示','判断与频率位置有关；后续结论配单因素对照，第二阶段计划频率随机平移训练。')]
    prose(s,left,35,101,430,244,12);prose(s,right,495,101,430,260,12)
    pic(s,'drone_table4.png',35,361,152,112);pic(s,'m100_window.png',200,361,265,112)
    pic(s,'move_focus.png',495,370,430,50);pic(s,'channel_before.png',495,431,203,50);pic(s,'channel_after.png',715,431,210,50)
    tx(s,40,475,425,18,'14 ms周期 / 约29%空窗',12,BLUE,True)
   elif i==6:
    rows=[['文献中的做法与发现','与本项目的关系','本项目评测原则'],['Soltani：7架同型号无人机；随机混合约97%，留出第四组约49%。DroneRFa原文随机划分。','同一录音相邻片段高度相似，不能作为独立测试。','按录音分组，训练与测试零泄漏。'],['WiSig跨设备识别发射机：单设备训练，换设备由99%降至33%以下。','两批实测数据来自不同设备，来源与类别相关。','分来源统计；检查设备杂散与底噪。'],['Geirhos综述无关线索：同分布测试难发现，需改变条件。','本项目平移实验100%→0–4%，需要控制位置与环境。','每项结论配单因素对照：平移、背景替换、特征打乱。']]
    table(s,rows,35,100,890,210,[365,270,255],12)
    pic(s,'soltani97.png',35,323,205,83);pic(s,'soltani49.png',250,323,205,83);pic(s,'wisig_fig10.png',495,320,430,127)
    tx(s,35,414,420,45,'原文两段分别给出随机混合与留出测试结果。仅改变划分，97%降至49%，说明相邻片段相似。',11)
    tx(s,495,454,430,30,'图10对比同设备与跨设备，设备多样性影响泛化。',10)
    tx(s,35,465,440,28,'归一化：WiSig防止功率成为位置信息；Glüge混合前也归一化。',10)
   elif i==7:
    prose(s,[C[i][0]],35,101,250,285,12)
    rows=[['项目','实验室数据','真实环境数据'],['环境','实验室内','室外及日常室内'],['设备与采样','一台；122.88 MS/s','另一台；重采样100 MS/s'],['内容','14型号，2.4/5.8 GHz，待机与飞行；无背景','7型号与环境背景，普遍有WiFi'],['规模','3933文件，61 GB','890段，73 GB']]
    table(s,rows,303,101,382,260,[63,159,160],11)
    tx(s,303,371,382,45,'共有型号：Mini 5 Pro、EVO II Pro V3，可用于同型号跨设备对比。',12)
    for y,label,val in [(125,'背景来源',100),(230,'2.4 GHz无人机来源',93)]:
     tx(s,710,y-24,215,24,label,11,BLUE,True);rect(s,710,y,215,24,PALE);rect(s,710,y,215*val/100,24,BLUE);tx(s,710,y+31,215,49,('100%真实环境' if val==100 else '93%实验室 / 7%真实环境'),12)
    tx(s,710,326,215,65,'背景 : 无人机\r2.4 GHz ≈ 1 : 13.8\r5.8 GHz ≈ 1 : 12.2',11)
    prose(s,[C[i][2]],35,415,650,76,11)
    tx(s,710,405,215,89,C[i][3][1],10,GRAY)
   elif i==8:
    prose(s,C[i],35,155,520,337,11.5)
    ops=['原始IQ','去杂散','截频切片','短时变换','功率池化','对数标准化']
    for k,t in enumerate(ops):
     node(s,35+k*150,102,137,37,t)
     if k<5:line(s,173+k*150,120,183+k*150,120)
    table(s,[['参数','2.4 GHz','5.8 GHz'],['采样率','61.44 MS/s','100 MS/s'],['频带MHz','2405–2455','5755–5845'],['短时变换','1024点窗','帧移512点'],['输出','1×256×256','1×256×256']],580,155,345,150,[95,125,125],10)
    pic(s,'window_miss.png',580,325,168,130);pic(s,'window_recall.png',758,325,167,130);tx(s,580,465,345,25,'窗长比较：漏检下降，但背景样本同步变少。',10)
   elif i==9:
    tx(s,35,98,890,54,'选择依据　ResNet18与DroneRFa时频图基线一致，便于延续比较。He提出把卷积输出与输入相加的残差连接，解决网络加深后训练困难的问题；本项目据此选用残差主干，手工特征分支的依据与筛选另在后页说明。',12)
    # Replace training note with causal explanations, keep all original hyperparameters.
    for sh in list(s.Shapes):
     if sh.HasTextFrame and sh.TextFrame.HasText and ('类别加权交叉熵' in sh.TextFrame.TextRange.Text):sh.Delete()
    tx(s,35,454,890,39,'类别加权（背景少）；按“误报＋漏检”选轮次（兼顾两类）；3次独立重复（排除偶然，方向一致才作为结论）。\rAdamW，学习率1×10⁻³，权重衰减1×10⁻⁴；40轮余弦退火至1×10⁻⁵；批大小64；混合精度。',11)
   elif i==10:
    prose(s,C[i],35,100,515,385,12)
    table(s,[['核验项','处理'],['97重复','剔除'],['15全零／25前段零','剔除／仅取有效段'],['相邻片段跨集合','按录音分组'],['采样率未记录','同步匹配并确认']],580,100,345,188,[175,170],11)
    pic(s,'line2426.png',580,315,345,105,'2426.88 MHz固定窄线；与采样率成整数比。');tx(s,580,460,345,29,'489块 / 173组 / 零泄漏',21,BLUE,True)
   elif i==11:
    generic(s,i);pic(s,'source_probe.png',600,102,325,160,'底噪预检验：不同来源检出差异明显。');pic(s,'background_clean.png',600,315,325,130);tx(s,600,450,325,32,'90%  vs  57%',23,BLUE,True);tx(s,600,482,325,13,'同一信号、0 dB，仅换背景',10)
   elif i==12:
    generic(s,i)
    rect(s,600,102,325,142,WHITE,True);tx(s,610,110,305,23,'取共轭 → 关于2400 MHz镜像',12,BLUE,True);node(s,610,150,140,50,'空闲段\r2342.6–2393.5');node(s,775,150,140,50,'工作频带\r2405–2455');line(s,750,175,774,175);tx(s,610,211,305,20,'单位MHz；背景随对应录音同组划分',10,GRAY)
    pic(s,'background_fa_clean.png',600,280,325,160);tx(s,600,457,325,32,'100% → 0%',23,BLUE,True)
   elif i==13:
    generic(s,i);pic(s,'gluge_mix3.png',600,100,325,78,'Glüge原文式(3)：先功率归一化，再按比例混合。');pic(s,'wifi_focus.png',600,250,325,180,'三种叠加做法；误差棒为3次训练标准差。');tx(s,600,463,325,29,'24%  vs  1.9%',23,BLUE,True)
   elif i==14:
    prose(s,C[i],35,100,420,388,11.5)
    rows=[['中文名','含义／计算','5特征'],['形状因子','幅度均方根／均值','否'],['幅度峰度（取对数）','四阶中心矩／方差²','否'],['峰均功率比','功率99.9%分位数，dB','是'],['归一化四阶累积量C42','E|z|⁴−|Ez²|²−2','否'],['瞬时幅度谱峰值（取对数）','归一化幅度傅里叶峰值','否'],['瞬时频率标准差','相位差分频率的标准差','是'],['谱平坦度','谱几何均值／算术均值','否'],['时频占用率','高于中位数10 dB比例','是'],['突发占空比','0.1 ms帧高于底噪6 dB','是'],['突发次数（取对数）','20 ms有信号起始次数','是']]
    table(s,rows,480,100,445,283,[182,211,52],10)
    for j,t in enumerate(['19 → 13：去6个功率／底噪类','13 → 10：去3个受IQ不平衡影响的量','10 → 5：设备检验，去1项明确与4项可疑']):
     node(s,480,394+j*30,445,26,t)
   elif i==15:
    generic(s,i);pic(s,'fusion.png',600,133,325,240,'柱与误差棒：3次训练均值、标准差。橙箭头连接置零与5特征柱顶。');tx(s,600,434,325,36,'94.9% → 86.2%',23,BLUE,True);tx(s,600,476,325,20,'5.8 GHz真实环境无人机检出',11)
   elif i==16:
    generic(s,i);pic(s,B/'normal.png',600,104,325,145,'正常分带示例：分别截取独立频带。');pic(s,B/'merged.png',600,315,325,145,'多信号合并示例：能量区域连成宽带。')
   elif i==17:
    table(s,[['指标','2.4 GHz 首轮→当前','5.8 GHz 首轮→当前'],['同设备背景误报','100%→0%','—'],['真实环境无人机检出','58%→85.7%','86%→85.4%'],['实验室无人机漏检','6.3%→5.1%','1.3%→1.3%'],['真实环境背景误报','0.4%→4.2%','0.6%→0%']],35,100,440,180,[170,135,135],11)
    prose(s,[C[i][1]],35,296,440,134,12)
    tx(s,35,444,440,43,'实验室漏检不含只有一段录音的型号：只能放训练或测试一边，无法按录音划分评估。',11)
    innovations=[('周期与位置先检验','常见基线以时频图分类；本项目先查完整周期、再平移，证据为14 ms空窗与100%→0–4%。'),('按录音划分与分来源','相邻片段随机划分会高估；本项目零泄漏并替换背景，检出90%对57%，区分环境影响。'),('构造背景与对称叠加','文献单录背景并数字混合；本项目空闲段镜像，误报100%→0%；再对两类叠加，误报24%对1.9%。'),('融合前检验设备信息','文献拼接深度与手工特征；本项目先查设备可区分性，谱平坦度0.934；融合后5.8 GHz检出下降。')]
    prose(s,innovations,500,100,425,326,12);tx(s,500,448,425,43,C[i][3][1],11,BLUE,True)
   elif i==18:
    rows=[['局限','影响','应对思路'],['测试背景仅80片','0误报的95%上界3.7%，无法证明≤1%','公开WiFi／蓝牙扩充；约300片无误报'],['5.8 GHz同设备无空闲背景','实验室只有无人机，特征可能区分来源','公开数据构造背景，先逐特征检验'],['2.4 GHz多信号交错','合并为60 MHz以上宽带，影响截取','按同时开关相关性切分，已知位置检验'],['实验室仅录至约2461 MHz','更高频图传缺失，处理无法恢复','列出数据边界，分频带报告'],['手工特征未稳定提升','Han融合部分条件有效，0 dB以下可能接近随机','低信噪比分档复评，再决定保留']]
    table(s,rows,35,101,890,278,[242,317,331],12)
    prose(s,[C[i][4]],35,397,890,90,12)
   elif i in [19,20]:
    columns=2 if i==19 else 3;ww=430 if columns==2 else 280;hh=184 if columns==2 else 181
    for k,item in enumerate(C[i]):
     x=35+(k%columns)*(460 if columns==2 else 305);y=102+(k//columns)*196
     prose(s,[item],x,y,ww,hh,11.5)
   if ROUND>=2 and i==3:
    for sh in list(s.Shapes):
     if sh.Left>=440 and sh.Top>=95 and sh.Top<490:sh.Delete()
    tx(s,445,99,480,25,'研究脉络：每一步的答案引出更深一层',14,BLUE,True)
    stages=[('一 能否识别','公开基线\r85.4%'),('二 是否可靠','平移后\r0–4%'),('三 判断依据','背景替换\r90%对57%'),('四 回到信号','检出率\r62%→81%'),('五 能否提升','融合未稳定\r分带进行中')]
    for k,(q,result) in enumerate(stages):
     xx=445+k*96;yy=275-k*32
     rect(s,xx,yy,90,116,WHITE,True);tx(s,xx+4,yy+8,82,40,q,12,BLUE,True);tx(s,xx+4,yy+58,82,50,result,11)
     if k<4:line(s,xx+90,yy+65,xx+96,yy+36);tx(s,xx+27,yy-25,100,20,'由此引出',10,GRAY)
    tx(s,445,420,480,65,'实测统一方法：核验与预处理 → 时频图＋手工特征 → 网络分类 → 单因素检验与改进。\r展望：低信噪比、中心频率输入、跨条件与未知型号拒识。',12)
   if ROUND>=2 and i==17:
    for sh in list(s.Shapes):
     if sh.Left<480 and sh.Top>=95 and sh.Top<495:sh.Delete()
    tx(s,35,100,440,26,'2.4 GHz：每步仅改变一个因素',13,BLUE,True)
    # Editable native evidence chart: no new measurement, no 85.7% fourth step.
    x0=75;y0=280;chartw=350;charth=140
    line(s,x0,132,x0,y0,False);line(s,x0,y0,450,y0,False)
    for v in [0,50,100]:
     yy=y0-v/100*charth;tx(s,40,yy-7,32,15,str(v)+'%',9,GRAY);line(s,x0-3,yy,x0+3,yy,False)
    xs=[100,245,415]
    for values,col in [([58,62,81],BLUE),([100,0,None],ORANGE)]:
     for k,v in enumerate(values):
      if v is None:continue
      yy=y0-v/100*charth;rect(s,xs[k]-2,yy-2,4,4,col);tx(s,xs[k]-23,yy-25,56,21,str(v)+'%',13,col,True)
      if k and values[k-1] is not None:
       sh=s.Shapes.AddLine(xs[k-1],y0-values[k-1]/100*charth,xs[k],yy);sh.Line.ForeColor.RGB=col;sh.Line.Weight=1
    for xx,label in zip(xs,['首轮','加入同设备背景','对称叠加干扰']):tx(s,xx-57,285,120,21,label,10)
    tx(s,35,313,440,35,'蓝：真实环境无人机检出；橙：同设备背景误报。\r对称叠加模拟误报1.9%（仅给无人机叠加为24%）。',10)
    tx(s,35,351,440,42,'当前基线重新训练3次：检出85.7%，真实环境背景误报4.2%。这不是新改进步骤。',11,BLUE,True)
    tx(s,35,404,440,74,'真实环境背景仅80片，0误报的95%上界仍为3.7%，尚不能证明≤1%。首轮低误报部分来自WiFi与背景的相关性，消除后需扩大背景统计。',12)
    # Rewrite innovation headings in the required layer order.
    for sh in s.Shapes:
     if sh.HasTextFrame and sh.TextFrame.HasText and sh.Left>=495:
      tr=sh.TextFrame.TextRange;t=tr.Text
      for old,new in [('周期与位置先检验','层次二／三：判断依据检验'),('按录音划分与分来源','层次四：同设备背景'),('构造背景与对称叠加','层次四：对称叠加'),('融合前检验设备信息','层次五：融合前检验来源')]:
       if t.startswith(old):
        bodies={'层次二／三：判断依据检验':'文献提示相邻片段与同分布高分存在局限；本项目按录音分组、分来源统计，并替换背景，证据为90%对57%。','层次四：同设备背景':'文献单录接收设备噪声；本项目在不新增采集条件下镜像空闲频段，并检验对侧泄漏，同设备背景误报100%降到0%。','层次四：对称叠加':'文献功率归一化后混合干扰；本项目增加两类同时叠加对照，检出62%升至81%，模拟误报24%降至1.9%。','层次五：融合前检验来源':'文献拼接深度与手工特征；本项目先查设备可区分性，谱平坦度0.934；5.8 GHz融合后下降，表明新输入需先验来源。'}
        tr.Text=new+'　'+bodies[new];style(tr,12);style(tr.Characters(1,len(new)),12,BLUE,True)
   if ROUND>=2 and i==18:
    tb=next(sh.Table for sh in s.Shapes if sh.HasTable)
    for k,layer in enumerate(['四','三／五','五','三','五'],2):
     tr=tb.Cell(k,1).Shape.TextFrame.TextRange;tr.Text='来自层次'+layer+'：'+tr.Text;style(tr,12)
   if ROUND>=2:progression(s,i)
  # Recompose every note, including retained template slides.
  for i,s in enumerate(p.Slides,1):
   s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text=notes(i)
   s.SlideShowTransition.EntryEffect=0
  print('Saving R4; waiting for embedded fonts',flush=True);p.SaveAs(str(OUT),24);print('saved',flush=True)
  p.Close()
 finally:
  try:p.Close()
  except:pass
  try:app.Quit()
  except:pass
 if before!=snapshot():raise RuntimeError('protected original changed')
 (NEW/'originals.json').write_text(json.dumps(before,ensure_ascii=False,indent=2),encoding='utf-8')

def walk(shapes):
 for sh in shapes:
  if sh.Type==6:yield from walk(sh.GroupItems)
  elif sh.HasTable:
   for r in range(1,sh.Table.Rows.Count+1):
    for c in range(1,sh.Table.Columns.Count+1):yield sh.Table.Cell(r,c).Shape
  else:yield sh
def audit():
 app=win32.DispatchEx('PowerPoint.Application');p=app.Presentations.Open(str(OUT),False,False,False);data=[]
 try:
  for i,s in enumerate(p.Slides,1):
   texts=[];fonts=[];overflow=[]
   for sh in walk(s.Shapes):
    if sh.HasTextFrame and sh.TextFrame.HasText:
     tr=sh.TextFrame.TextRange;texts.append(tr.Text)
     for k in range(1,tr.Runs().Count+1):
      fs=float(tr.Runs(k).Font.Size)
      if fs>0:fonts.append(fs)
     if tr.BoundHeight>sh.Height+3 or tr.BoundWidth>sh.Width+3:
      overflow.append({'text':tr.Text,'rect':[float(sh.Width),float(sh.Height)],'bound':[tr.BoundWidth,tr.BoundHeight]})
   note=s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text
   data.append({'page':i,'chars':sum(len(re.sub(r'\s','',t)) for t in texts),'body_chars':sum(len(t) for h,t in C.get(i,[])),'min_font':min(fonts) if fonts else None,'notes_chars':len(note),'texts':texts,'notes':note,'overflow':overflow})
   s.Export(str(P/f'r4_round{ROUND}_{i:02d}.png'),'PNG',1920,1080);s.Export(str(P/f'r4_{i:02d}.png'),'PNG',1920,1080)
   print('export',ROUND,i,flush=True)
 finally:
  p.Close()
  try:app.Quit()
  except:pass
 (NEW/f'audit_round{ROUND}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 files=sorted(P.glob(f'r4_round{ROUND}_??.png'));im=Image.new('RGB',(1920,1740),'#dfe5ea');dr=ImageDraw.Draw(im)
 for k,path in enumerate(files):
  x=k%4*480;y=k//4*290;im.paste(Image.open(path).resize((480,270)),(x,y));dr.text((x+8,y+271),str(k+1),fill='black')
 im.save(D/'contact_r4.png');im.save(NEW/f'contact_round{ROUND}.png')
 print('OVERFLOWS',[(d['page'],len(d['overflow'])) for d in data if d['overflow']],flush=True)
if __name__=='__main__':
 if '--audit' not in sys.argv:build()
 audit()
