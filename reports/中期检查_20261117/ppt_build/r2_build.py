from pathlib import Path
import os,sys,json,re,math,shutil
from PIL import Image,ImageDraw
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
B=Path(__file__).resolve().parent;DROOT=B.parent;ROOT=B.parents[2];A=B/'r2_assets';P=DROOT/'ppt_preview'
ROUND=int(sys.argv[1]) if len(sys.argv)>1 else 1
base=(DROOT/'build_ppt.py').read_text(encoding='utf-8-sig').split('# ================= 封面、目录')[0]
base=base.replace("D = os.path.dirname(os.path.abspath(__file__))","D = str(DROOT)")
base=base.replace('pres.SaveAs(OUT)','app.DisplayAlerts=1')
base=base.replace("S = {name: pres.Slides.FindBySlideID(sid) for name, sid in ids.items()}","S = {name:pres.Slides(j) for j,name in enumerate(keep.values(),1)}")
exec(base,globals());exec((B/'r2_content.py').read_text(encoding='utf-8'),globals())
Q[13]=Q[13][:3]+[('⑤ 结果与留下的问题',Q[13][3][1]+Q[13][4][1])]
Q[19]=[Q[19][0],Q[19][2],Q[19][1],Q[19][3]]
OUT=str(B/f'r2_build_{os.getpid()}.pptx')
slides=[S['cover'],S['toc']];stats=[];body_records={};image_records={}
task=(ROOT/'work/codex_tasks/PPT_TASK_R2_内容重写_逐页内容稿_20261117.md').read_text(encoding='utf-8-sig')
parts=re.split(r'### 第 (\d+) 页：',task);notes={}
for k in range(1,len(parts),2):notes[int(parts[k])]=parts[k+1].split('- **配图**')[0].split('- 配图：')[0].replace('**','').replace('---','').strip()
def tx(s,x,y,w,h,t,size=13,c=TXT,b=False,align=1):
 size=max(10,size)
 sh=s.Shapes.AddTextbox(1,x,y,w,h);tf=sh.TextFrame;tf.TextRange.Text=t;tf.WordWrap=-1;tf.AutoSize=0
 tf.MarginLeft=tf.MarginRight=1;tf.MarginTop=tf.MarginBottom=1;tf.VerticalAnchor=1
 style(tf.TextRange,size,c,b);tf.TextRange.ParagraphFormat.Alignment=align;tf.TextRange.ParagraphFormat.SpaceAfter=0
 return sh
def rect(s,x,y,w,h,c=LGRAY,line=None,round=False):
 sh=s.Shapes.AddShape(5 if round else 1,x,y,w,h);sh.Fill.Solid();sh.Fill.ForeColor.RGB=c;sh.Line.Visible=0 if line is None else -1;sh.Shadow.Visible=0
 if line is not None:sh.Line.ForeColor.RGB=line;sh.Line.Weight=.75
 if round:sh.Adjustments.SetItem(1,.06)
 return sh
def outline(s,x,y,w,h,c=ORANGE):
 sh=rect(s,x,y,w,h,WHITE,c,True);sh.Fill.Visible=0;sh.Line.Weight=1.5;return sh
def pic(s,path,x,y,w,h,page=None):
 path=Path(path);path=path if path.is_absolute() else A/path
 im=Image.open(path);r=min(w/im.width,h/im.height);pw=im.width*r;ph=im.height*r
 sh=s.Shapes.AddPicture(str(path),0,-1,x+(w-pw)/2,y+(h-ph)/2,pw,ph);sh.Line.Visible=-1;sh.Line.Weight=.75;sh.Line.ForeColor.RGB=rgb(215,222,228)
 if page:image_records.setdefault(page,[]).append(str(path))
 return sh
def title2(s,t):
 sh=s.Shapes.Title;sh.TextFrame.TextRange.Text=t;sh.Left=35;sh.Top=15;sh.Width=748;sh.Height=43;style(sh.TextFrame.TextRange,26,TXT,False,'方正大标宋简体')
def add(page):
 s=new_slide(TITLES[page]);title2(s,TITLES[page]);slides.append(s);set_notes(s,notes.get(page,''));print('R2 page',page,flush=True);return s
def tag(s,t):rect(s,803,77,122,22,rgb(232,242,250),None,True);tx(s,805,80,118,19,t,10.5,DBLUE,True,2)
def banner2(s,t,y=462):rect(s,35,y,890,35,rgb(232,242,250),None,True);tx(s,46,y+6,868,27,t,13,DBLUE,True)
def cap(s,t,x=584,y=460,w=341,h=35):tx(s,x,y,w,h,t,10,GRAY)
def groups(s,page,items,x=35,y=87,w=523,h=407):
 body_records[page]=items;objects=[]
 for head,t in items:
  hs=tx(s,x+13,y,w-13,22,head,13.5,BLUE,True);bs=tx(s,x+13,y+23,w-13,400,t,12.6)
  objects.append((hs,bs))
 heights=[23+o[1].TextFrame.TextRange.BoundHeight+12 for o in objects]
 if sum(heights)>h:
  for _,bs in objects:bs.TextFrame.TextRange.Font.Size=12
  heights=[23+o[1].TextFrame.TextRange.BoundHeight+12 for o in objects]
 if sum(heights)>h+3:print('GROUP HEIGHT WARNING',page,sum(heights),h,flush=True)
 yy=y
 for (hs,bs),hh in zip(objects,heights):
  hs.Top=yy;bs.Top=yy+23;bs.Height=bs.TextFrame.TextRange.BoundHeight+2;rect(s,x,yy+4,5,13,BLUE);yy+=hh
def table(s,rows,x,y,w,h,widths,size=12):
 sh=s.Shapes.AddTable(len(rows),len(rows[0]),x,y,w,h);tb=sh.Table
 for j,cw in enumerate(widths,1):tb.Columns(j).Width=cw
 for i,row in enumerate(rows,1):
  for j,t in enumerate(row,1):
   c=tb.Cell(i,j).Shape;c.TextFrame.TextRange.Text=t;style(c.TextFrame.TextRange,size,WHITE if i==1 else TXT,i==1)
   c.TextFrame.MarginLeft=c.TextFrame.MarginRight=5;c.TextFrame.MarginTop=c.TextFrame.MarginBottom=3
   c.Fill.Solid();c.Fill.ForeColor.RGB=BLUE if i==1 else (rgb(232,242,250) if j==len(row) else WHITE)
 return sh
def trans(page,n,part,t,sub):
 s=transition(n,part,t,sub,notes.get(page,''));slides.append(s)
 sh=shape_by_name(s,'文本占位符 12');sh.Width=835;sh.Height=55;sh.TextFrame.TextRange.Font.Size=30
 for sh in s.Shapes:
  if sh.HasTextFrame and sh.TextFrame.HasText and sh.TextFrame.TextRange.Text==n:sh.Height=225
 return s
try:
 shape_by_name(S['cover'],'标题 1').TextFrame.TextRange.Text='基于深度学习的无人机\r射频信号检测识别'
 shape_by_name(S['cover'],'文本占位符 85').TextFrame.TextRange.Text='大学生创新训练计划（SRTP）中期检查答辩'
 shape_by_name(S['cover'],'文本占位符 87').TextFrame.TextRange.Text='指导教师：xxx　　汇报人：xxx'
 set_notes(S['cover'],'各位老师好。本阶段围绕模型凭什么判断开展工作。我将说明每个问题怎样被发现、证实和修正，以及还留下什么问题。')
 shape_by_name(S['toc'],'文本占位符 4').TextFrame.TextRange.Text='研究目标与评测思路\r公开数据集阶段：第一次发现模型“走捷径”\r实测数据阶段：逐个发现并排除捷径\r阶段成果、创新点与不足'
 style(shape_by_name(S['toc'],'文本占位符 4').TextFrame.TextRange,25,BLUE,False,'方正大标宋简体')
 set_notes(S['toc'],'按照目标、公开数据集、实测数据、阶段成果四部分汇报。重点是中间的问题链，计划只在最后一页简要说明。')
 s=add(3)
 rect(s,35,89,427,324,WHITE,BLUE,True);rect(s,480,89,445,324,WHITE,ORANGE,True)
 tx(s,52,106,390,32,'任务：两阶段识别',20,BLUE,True)
 tx(s,52,153,390,235,'第一阶段：判断有无无人机\r第二阶段：识别具体型号\r\r各机型漏检 ≤1%，背景虚警 ≤1%\r不加噪、0、−5、−10、−15 dB\r平均准确率 ≥95%',16)
 tx(s,499,106,405,32,'难点：录音来源与标签绑定',20,ORANGE,True)
 tx(s,499,153,405,235,'不同接收机、不同环境录制的数据，常把来源与标签绑在一起。\r\r模型可能靠接收机、环境或频率位置拿高分，普通测试集看不出，换场景就失效。',17)
 banner2(s,'本阶段主线：每得到一个结果，都追问判断依据，并设计能推翻结论的对照实验',443)
 s=add(4)
 rows=[['主流做法','文献暴露的问题','我们的做法'],['时频图 + CNN\rDroneRFa / Glüge','短片段不能观测完整统计特征','ResNet18基线；比较10–50 ms'],['随机切分窗口\rDroneRFa 6:2:2 / Ezuma 80/20','Soltani：随机97%，留出突发49%','按录音分组，训练/测试零泄漏'],['单接收机、总体准确率','WiSig换接收机99%→<33%\rGeirhos：同分布测试藏住捷径','分来源统计；替换线索对照'],['去掉功率信息','功率随距离和增益变，暴露位置','先功率归一化；绝对功率作阳性对照']]
 table(s,rows,35,86,890,184,[231,330,329],12)
 pic(s,'soltani97.png',35,286,215,85,4);pic(s,'soltani49.png',262,286,215,85,4)
 tx(s,35,381,442,36,'同一任务，随机切分97% → 跨突发49%。\r因此用按录音分组替代随机窗口划分。',12,DBLUE,True)
 cap(s,'图源：Soltani, IEEE TVT 2020, Sec.III-D, p.4–5（原文证据）',35,425,442,27)
 pic(s,'wisig_fig10.png',495,280,430,155,4)
 cap(s,'图源：Hanna, WiSig arXiv 2022, Fig.10, p.6。换接收机99%→<33%，须做留出测试。',495,439,430,37)
 tx(s,35,480,890,20,'调研：高权威34篇（精读10篇）· 融合19篇（全文12篇）· 公开WiFi/蓝牙数据集21个',10.2,GRAY)
 s=add(5)
 nodes=[('周期性漏检','长时时频图','14 ms周期；空窗29%'),('M300仅34.8%','频率平移对照','100% → 0–4%'),('数据不干净','全面审核','处理137个文件'),('本振谱线','逐频点统计','定位5条稳定线'),('来源标签绑定','只看底噪分类','0%虚警、93%检出'),('模型依据什么','背景替换','90% → 57%'),('缺同接收机背景','空闲频段镜像','100% → 0%'),('WiFi怎么补','对称干扰叠加','24% vs 1.9%'),('IQ特征有用吗','先验来源再融合','无稳定提升 / 下降')]
 rect(s,35,85,890,116,rgb(241,247,251),BLUE,True);tx(s,45,89,200,22,'公开数据集：问题链1–2',11,DBLUE,True)
 rect(s,35,212,890,244,rgb(248,249,251),BLUE,True);tx(s,45,217,200,22,'实测数据：问题链3–9',11,DBLUE,True)
 positions=[(55,116),(345,116),(55,246),(345,246),(635,246),(635,351),(345,351),(55,351),(635,116)]
 # Chronological 1–2 at top, then 3–9 in a two-row snake. Node9 is placed at the end of bottom row.
 positions=[(55,116),(345,116),(55,245),(273,245),(491,245),(709,245),(709,351),(491,351),(273,351)]
 for j,((h,m,r),(x,y)) in enumerate(zip(nodes,positions)):
  w=270 if j<2 else 198;rect(s,x,y,w,83,WHITE,BLUE,True);tx(s,x+8,y+6,w-16,21,f'{j+1}  {h}',13,ORANGE,True);tx(s,x+8,y+31,w-16,21,m,12,BLUE);tx(s,x+8,y+55,w-16,22,r,12,DBLUE,True)
  if j in [0,2,3,4]:tx(s,x+w+2,y+32,18,23,'→',14,GRAY)
  if j in [6,7]:tx(s,x-20,y+32,18,23,'←',14,GRAY)
  if j in [0,2,3,4]:tx(s,x+w+1,y+60,25,17,'引出',8,GRAY)
  if j in [6,7]:tx(s,x-23,y+60,25,17,'引出',8,GRAY)
 # Carry the chain between the two dataset bands and around the snake turn.
 line=s.Shapes.AddLine(635,157,895,157);line.Line.ForeColor.RGB=GRAY
 line=s.Shapes.AddLine(895,157,895,223);line.Line.ForeColor.RGB=GRAY
 line=s.Shapes.AddLine(895,223,152,223);line.Line.ForeColor.RGB=GRAY;line.Line.EndArrowheadStyle=3
 tx(s,661,160,200,19,'引出实测数据审核',10,GRAY)
 tx(s,799,331,25,20,'↓',14,GRAY);tx(s,826,332,40,18,'引出',9,GRAY)
 rect(s,55,351,198,83,rgb(242,245,248),GRAY,True);tx(s,64,367,180,52,'进行中\r按信号所在频带截取',13,GRAY,True,2)
 tx(s,35,472,890,26,'每个节点的结果暴露下一条线索，后续页面按问题链逐项展开。',13,DBLUE)
 trans(6,'贰','第二部分','公开数据集阶段：第一次发现“捷径”',['周期性漏检','位置捷径'])
 s=add(7);tag(s,'问题链 1/9');groups(s,7,Q[7])
 pic(s,'drone_table4.png',584,110,341,182,7);cap(s,'图源：俞宁宁等,《电子与信息学报》2024, 表4, p.1153（PDF p.7）。先看周期，再定窗长。',584,298,341,40)
 pic(s,'m100_window.png',584,344,341,110,7);cap(s,'本项目：Matrice 100，15 ms输入示例。短突发之外是长间隙，未把弱纹理当作强突发。',584,460,341,36)
 outline(s,739,359,155,69)
 s=add(8);tag(s,'问题链 2/9');groups(s,8,Q[8]);pic(s,'move_focus.png',584,140,341,66,8)
 tx(s,584,215,113,23,'不移动',12,DBLUE,True,2);tx(s,697,215,113,23,'平移方案1',12,DBLUE,True,2);tx(s,810,215,115,23,'平移方案2',12,DBLUE,True,2)
 cap(s,'M300验证片段：形状不变，仅挪频率，100%→4%/0%。图源：本项目06图第一行前三个单元格。',584,250,341,59)
 pic(s,'channel_before.png',584,336,164,91,8);pic(s,'channel_after.png',760,336,165,91,8)
 cap(s,'本项目M300 S0110：换信道前（左）/后（右）。图传中心由约2456跳至2409 MHz，与误判时间吻合。',584,434,341,53)
 trans(9,'叁','第三部分','实测数据阶段：逐个发现并排除捷径',['数据审核 / 谱线 / 来源混杂','WiFi捷径 / 同接收机背景 / 对称叠加 / 特征融合'])
 s=add(10);tag(s,'问题链 3/9')
 rows=[['发现的问题','怎么查出来的','不处理的后果','处理'],['97个重复（1.56 GB）','逐文件MD5；核查(1)与时间戳后缀','同段信号跨训练/测试','核验剔除'],['15全零 + 25前段零\r均为5.8 GHz第0轮','逐文件统计零值长度','空样本被标为无人机','全零剔除；前段零仅取有效区间'],['原始划分泄漏','同录音seg00–17训练、seg18–19测试','近邻相似，成绩虚高\rSoltani：97%→49%','按录音分组'],['采样率缺失','6,144,000点/50 ms；同步序列匹配\r10/20/40 MHz实测约10/19/37','无法换算时长、频率','推断122.88 MS/s，经确认']]
 table(s,rows,35,114,890,247,[205,286,205,194],12)
 tx(s,35,381,890,60,'处理137个问题文件：剔除112个，25个只取有效区间。\r切片得到489个信号块、173组，复核训练集与测试集零泄漏。',15,DBLUE,True)
 cap(s,'数据规模：实验室3933文件 / 61 GB；城区与家居890段 / 73 GB；v3共1658文件 / 120 GB。',35,467,890,26)
 body_records[10]=[('审核表',''.join(''.join(r) for r in rows[1:]))]
 s=add(11);tag(s,'问题链 4/9');groups(s,11,Q[11]);pic(s,'line2426.png',584,120,341,113,11);outline(s,744,149,24,67)
 cap(s,'2426.88 MHz统计候选位置（红线）。图源：本项目lines_data_2.4GHz_page10，第13子图。是否稳定由逐录音出现率判定。',584,242,341,58)
 rect(s,584,318,341,131,rgb(242,245,248),None,True);tx(s,598,329,313,107,'为什么用同一陷波表？\r若只处理某一来源，模型还会学到“有没有凹坑”。\r\r2.4 GHz：2426.88 / 2440 / 2450\r5.8 GHz：5795.5 / 5800 MHz',12,DBLUE)
 s=add(12);tag(s,'问题链 5/9');groups(s,12,Q[12]);pic(s,'source_probe.png',584,113,341,226,12)
 cap(s,'只看底噪、不看无人机：实验室检出高，城区几乎全漏。图源：R2第12页给定结果重绘；同来源对照平衡准确率0.51。',584,350,341,55)
 tx(s,595,421,318,70,'0%虚警 + 93.2%检出\r也可能完全没有学到无人机。',20,ORANGE,True)
 s=add(13);tag(s,'问题链 6/9');groups(s,13,Q[13]);pic(s,B/'background.png',584,122,341,218,13);outline(s,649,170,28,71)
 cap(s,'左图0 dB处白噪声约90%、城区约57%；右图城区样本加白噪反而更易检出。说明环境线索影响判断。图源：本项目图B上两子图。',584,351,341,74)
 tx(s,596,428,317,36,'90%  vs  57%',24,ORANGE,True,2)
 tx(s,596,471,317,23,'同一信号，只换背景就变了',12,DBLUE,True,2)
 s=add(14);tag(s,'问题链 7/9');Q[14][1]=(Q[14][1][0],Q[14][1][1].replace('；','；\r').replace('。原因','。\r原因'));groups(s,14,Q[14])
 tx(s,584,108,341,23,'共轭镜像：保留同接收机底噪',14,BLUE,True)
 line=s.Shapes.AddLine(594,194,914,194);line.Line.ForeColor.RGB=BLUE;line.Line.EndArrowheadStyle=3
 rect(s,598,164,128,19,BLUE);rect(s,780,164,130,19,DBLUE);tx(s,596,207,139,46,'空闲段\r2342.6–2393.5 MHz',10.5,BLUE,False,2);tx(s,774,207,142,46,'工作频带\r2405–2455 MHz',10.5,DBLUE,False,2)
 tx(s,735,178,37,30,'fc',12,TXT,False,2);arc=s.Shapes.AddShape(25,647,136,190,44);arc.Fill.Visible=0;arc.Line.ForeColor.RGB=BLUE;arc.Line.EndArrowheadStyle=3
 pic(s,'bgA_fa.png',584,271,341,181,14);cap(s,'图源：本项目图C左子图。重点看A背景柱：补入镜像背景后，虚警从100%降为0%。',584,457,341,39)
 s=add(15);tag(s,'问题链 8/9');groups(s,15,Q[15]);pic(s,'gluge_mix3.png',584,115,341,92,15)
 cap(s,'图源：Glüge, IEEE J-RFID 2024, 式(3), p.3。先归一化，再按SNR混合。',584,217,341,38)
 pic(s,'wifi_focus.png',584,269,341,204,15)
 cap(s,'图源：metrics_paired.csv，J/La/Lb。La虚警24%，Lb仅1.9%，须兼看召回。',584,474,341,24)
 s=add(16)
 pic(s,'window_miss.png',35,84,433,217,16);pic(s,'window_recall.png',492,84,433,217,16)
 tx(s,35,312,890,65,'设计：10/20/30/40/50 ms × 两频段 × 3种子，只改窗长。5.8 GHz漏检1.92%→1.28%→0.24%，城区召回81.8%→85.9%→93.1%（10/20/30 ms）。2.4 GHz在30 ms时漏检1.96%、召回92.4%，20 ms为5.70%、86.0%。',13)
 tx(s,35,385,890,70,'但不能据此选定长窗：2.4 GHz背景由10 ms的180片降为40 ms的40片，“0次虚警”的95%上界由1.65%放宽至7.22%；每列时间分辨率也由0.04 ms降至0.20 ms。背景数量和统计把握随窗长一起变化，虚警不能直接横比。',13)
 cap(s,'图源：metrics_summary.csv，argmax，均值±标准差；各点N为对应频段测试背景片数。窗长选择仍受背景不足约束。',35,466,890,28)
 s=add(17);tag(s,'问题链 9/9（上）')
 rows=[['文献','做法','给我们的约束'],['Open-RFNet\rJSAC 2026稿','嵌入拼接后接MLP','融合0.994未超过ResNet 0.9958'],['Han / FuseAMC','深度 + 累积量/统计量','≤0 dB手工0.24 vs CNN 0.34'],['Glüge / WiSig / Shebert','功率归一化','不能靠功率当位置线索'],['Wang, CVPR 2020','多模态融合研究','与最好单模态公平比较']]
 table(s,rows,35,110,523,154,[115,175,233],12)
 items=[('③ 依据','导师建议加入IQ特征，先调研19篇、读全文12篇。文献有嵌入拼接先例，却也显示强分支会主导梯度、低信噪比手工特征失效，融合未必优于单分支。'),('④ 怎么解决的','归一化后计算19特征；ResNet18的512维嵌入与MLP投影64维拼接，设置零组Z。用A/B内容相同镜像背景验来源：谱平坦度AUC 0.934，剔除；功率与底噪阳性对照归一化后AUC 0.50。保留10特征F，其中5个无差异为F5。'),('⑤ 结果与留下的问题','设计包含来源检验和公平对照。→ 特征加入后究竟提高了谁的指标？')]
 groups(s,17,items,y=280,h=216)
 pic(s,'openrf_concat.png',584,113,341,152,17);outline(s,771,163,29,48)
 cap(s,'图源：Gao等，JSAC 2026稿，Fig.2局部，p.5。\rC为拼接，再送入网络头（非加法⊕）。',584,275,341,43)
 pic(s,'mirror_feature.png',584,336,341,124,17);cap(s,'图源：本项目图17的谱平坦度箱图。对内容相同背景仍能分接收机，需剔除该特征。',584,465,341,34)
 s=add(18);tag(s,'问题链 9/9（下）');groups(s,18,Q[18]);pic(s,B/'fusion_focus.png',584,113,341,293,18)
 cap(s,'上：F虚警均值较低，但种子方向不一致。\r下：Z→F→F5召回下降。\r图源：metrics_summary.csv，均值±标准差。',584,414,341,71)
 line=s.Shapes.AddLine(757,346,866,374);line.Line.ForeColor.RGB=ORANGE;line.Line.Weight=2;line.Line.EndArrowheadStyle=3
 s=add(19);tag(s,'进行中');groups(s,19,Q[19]);pic(s,B/'normal.png',584,116,341,130,19);pic(s,B/'merged.png',584,286,341,130,19);outline(s,619,310,291,74)
 cap(s,'正常：实验室能分开多条频带。图源：k6_g2_data_drone_01的Q0000。',584,250,341,32)
 cap(s,'失败：城区背景被并为宽带。图源：k6_g2_raw_background_01的Q1026。示例用于说明合并现象，非60–68 MHz同一条录音。',584,426,341,61)
 trans(20,'肆','第四部分','阶段成果、创新点与不足',[])
 s=add(21)
 rows=[['指标','2.4首轮','2.4当前','5.8首轮','5.8当前'],['A背景虚警','100%','0%','—','—'],['城区召回','58%','85.7%','86%','85.4%'],['实验室漏检\r不含泛化集','6.3%','5.1%','1.3%','1.3%'],['城区背景虚警','0.4%','4.2%','0.6%','0%']]
 table(s,rows,35,93,523,201,[155,92,92,92,92],12.5)
 tx(s,35,309,523,116,'2.4 GHz虚警变高也必须如实报告。第一轮低虚警部分依赖“有WiFi=背景”；去掉捷径后，80片约3片误报，暴露了仍需解决的真实水平。\r\r当前基线：B2b的R组，对称叠加重训，3种子均值。',14)
 tx(s,584,93,341,33,'方法与积累',20,BLUE,True)
 tx(s,584,142,341,328,'完整流程\r审核、去谱线、截频、切片、分组划分，\r训练后做对照检验。\r\r捷径检验\r背景替换、白噪声/陷波噪声、分来源，\r同型号对照、事先约定判读规则。\r\r每组3种子，报告95%置信上界。\r约360 GB；11月60余次训练。\r26份任务书、20份报告、100余张图。',13.5)
 s=add(22)
 data=[('实测数据的捷径检验流程','文献多报随机划分、单接收机总体准确率。我们做背景替换、陷波噪声、分来源及同型号对照。证据：90% vs 57%，空频段误报100%。'),('不补录构造同接收机背景','Glüge另录接收机底噪。我们利用同录音空闲频段镜像，并检验泄漏。证据：虚警100%→0%。'),('证明干扰需要对称叠加','Glüge混背景与无人机。我们在接收机来源不同条件下比较不对称/对称叠加。证据：虚警24% vs 1.9%。'),('先检验来源，再做IQ特征融合','融合文献未检验来源指纹。我们用镜像背景检验，剔除谱平坦度（AUC 0.934）。5.8 GHz召回下降反证来源风险。')]
 for j,(h,t) in enumerate(data):
  x=35+j%2*455;y=90+j//2*182;rect(s,x,y,435,162,WHITE,BLUE,True);tx(s,x+14,y+12,407,33,h,17,BLUE,True);tx(s,x+14,y+56,407,98,t,13)
 cap(s,'据目前检索，以上差异和文献覆盖范围需在答辩前再核对。',35,473,890,24)
 s=add(23)
 tx(s,35,95,523,32,'尚未解决的四个问题',19,ORANGE,True)
 tx(s,35,144,523,300,'① 背景仅80片：0/80的95%上界3.7%，无法验证虚警≤1%。\r\r② 5.8 GHz没有接收机A背景，来源与标签仍绑定。\r\r③ 城区2.4 GHz频带切分未解决。\r\r④ A只录到约2461 MHz，更高频图传缺失。',16)
 tx(s,594,95,331,32,'下一步只围绕这些缺口',19,BLUE,True)
 tx(s,594,144,331,300,'① 公开WiFi/蓝牙对称叠加，扩充有干扰背景。\r\r② 按信号频带截取后重训。\r\r③ 低信噪比训练与第二阶段型号识别。',16)
 cap(s,'结题时间待核对。',35,473,890,24)
 e=S['end'];slides.append(e)
 for sh in e.Shapes:
  if sh.HasTextFrame and sh.TextFrame.HasText:
   t=sh.TextFrame.TextRange.Text
   if '指导教师' in t:sh.TextFrame.TextRange.Text='指导教师：xxx'
   if '答辩学生' in t or '汇报人' in t:sh.TextFrame.TextRange.Text='汇报人：xxx'
 set_notes(e,'谢谢各位老师。请对模型依据的检验、当前基线及下一步重点批评指正。')
 S['trans'].Delete()
 for i,s in enumerate(slides,1):s.MoveTo(i)
 for page,value,label in [(7,'14 ms周期 / 约29%空窗','先解释漏检，再决定窗口'),(8,'100% → 0–4%','不重新训练，只改变频率位置'),(11,'5 条稳定谱线','来源特有的窄线不能成为标签'),(14,'100% → 0%','同接收机背景虚警；WiFi仍待解决'),(15,'24%  vs  1.9%','不对称叠加 / 对称叠加'),(18,'94.9% → 89.2% → 86.2%','5.8 GHz城区召回：Z / F / F5')]:
  ss=slides[page-1];tx(ss,48,430,510,37,value,24,ORANGE if page in [8,15] else DBLUE,True)
  tx(ss,48,474,510,23,label,12,GRAY)
 bad=['请在此','这是此模版','单击添加','关键词','学校肇始','芃苇','PengV','Page']
 def walk(shapes):
  for sh in shapes:
   if sh.Type==6:yield from walk(sh.GroupItems)
   else:yield sh
 for des in pres.Designs:
  for container in [des.SlideMaster]+list(des.SlideMaster.CustomLayouts):
   for sh in walk(container.Shapes):
    if sh.HasTextFrame and sh.TextFrame.HasText and any(b in sh.TextFrame.TextRange.Text for b in bad):sh.TextFrame.TextRange.Text=''
 for i,s in enumerate(slides,1):
  for sh in list(s.Shapes):
   if sh.Top<0 or sh.Top>=540:sh.Delete();continue
   if sh.HasTextFrame and sh.TextFrame.HasText and any(b in sh.TextFrame.TextRange.Text for b in bad):sh.TextFrame.TextRange.Text=''
  s.SlideShowTransition.EntryEffect=0
  while s.TimeLine.MainSequence.Count:
   try:s.TimeLine.MainSequence.Item(1).Delete()
   except:break
  if i not in [1,2,6,9,20,24]:tx(s,894,513,30,18,str(i),10,GRAY,False,3)
 print('Saving R2',OUT,flush=True);pres.SaveAs(OUT,24);print('Saved R2, exporting',flush=True)
 for i,s in enumerate(slides,1):
  s.Export(str(P/f'r2_round{ROUND}_{i:02d}.png'),'PNG',1920,1080);s.Export(str(P/f'r2_{i:02d}.png'),'PNG',1920,1080)
  ts=[];fonts=[];ov=[]
  def shapes_all(shapes):
   for sh in shapes:
    if sh.Type==6:yield from shapes_all(sh.GroupItems)
    elif sh.HasTable:
     for r in range(1,sh.Table.Rows.Count+1):
      for c in range(1,sh.Table.Columns.Count+1):yield sh.Table.Cell(r,c).Shape
    else:yield sh
  for sh in shapes_all(s.Shapes):
   if sh.HasTextFrame and sh.TextFrame.HasText:
    tr=sh.TextFrame.TextRange;ts.append(tr.Text)
    try:fonts.append(float(tr.Font.Size))
    except:pass
    try:
     if tr.BoundHeight>sh.Height+3 or tr.BoundWidth>sh.Width+3:ov.append([sh.Name,tr.Text,tr.BoundHeight,sh.Height,tr.BoundWidth,sh.Width])
    except:pass
  stats.append({'page':i,'texts':ts,'chars':sum(len(re.sub(r'\s','',t)) for t in ts),'min_font':min([v for v in fonts if v>0]) if fonts else None,'body_chars':sum(len(t) for _,t in body_records.get(i,[])),'groups':body_records.get(i,[]),'overflow':ov,'images':image_records.get(i,[])})
 pres.Close()
 try:app.Quit()
 except:pass
 shutil.copy2(OUT,DROOT/'中期答辩_模板稿.pptx')
 data={'slides':stats,'round':ROUND,'build':OUT,'residual':{b:sum(t.count(b) for x in stats for t in x['texts']) for b in bad}}
 (B/f'r2_audit_round{ROUND}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 contact=Image.new('RGB',(1920,1740),'#dfe5ea');dr=ImageDraw.Draw(contact)
 for i in range(1,25):
  im=Image.open(P/f'r2_{i:02d}.png');im.resize((960,540)).save(P/f'r2_review_{i:02d}.png');x=(i-1)%4*480;y=(i-1)//4*290;contact.paste(im.resize((480,270)),(x,y));dr.text((x+8,y+270),str(i),fill='black')
 contact.save(P/'contact_r2.png');contact.save(P/f'contact_r2_round{ROUND}.png')
 print('R2 done 24 pages',[(r['page'],len(r['overflow'])) for r in stats if r['overflow']],flush=True)
finally:
 try:pres.Saved=True;pres.Close()
 except:pass
 try:app.Quit()
 except:pass
