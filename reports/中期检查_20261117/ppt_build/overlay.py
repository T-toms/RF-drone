from PIL import Image,ImageDraw
import json,re,hashlib
B=Path(__file__).resolve().parent
P=B.parent/'ppt_preview';P.mkdir(exist_ok=True)
# Capture native template elements for reuse before removing donors.
def walk(shapes):
 for sh in shapes:
  if sh.Type==6:
   yield from walk(sh.GroupItems)
  else: yield sh
def wipe(s):
 keepname=s.Shapes.Title.Name if s.Shapes.Title is not None else None
 for k in range(s.Shapes.Count,0,-1):
  sh=s.Shapes(k)
  if sh.Name!=keepname and (sh.Top>=70 or sh.Top<0):sh.Delete()
def tx(s,x,y,w,h,msg,size=16,c=TXT,b=False,align=1):
 sh=s.Shapes.AddTextbox(1,x,y,w,h);tf=sh.TextFrame
 tf.TextRange.Text=msg;tf.WordWrap=-1;tf.AutoSize=0
 tf.MarginLeft=tf.MarginRight=2;tf.MarginTop=tf.MarginBottom=2
 tf.VerticalAnchor=1;style(tf.TextRange,size,c,b)
 tf.TextRange.ParagraphFormat.Alignment=align;tf.TextRange.ParagraphFormat.SpaceAfter=0
 return sh
def box(slide,l,t,w,h,fill=LGRAY,line=None,radius=True):
 sh=slide.Shapes.AddShape(5 if radius else 1,l,t,w,h);sh.Fill.Solid();sh.Fill.ForeColor.RGB=fill
 sh.Line.Visible=0 if line is None else -1
 if line is not None:sh.Line.ForeColor.RGB=line;sh.Line.Weight=.75
 sh.Shadow.Visible=0
 if radius:sh.Adjustments.SetItem(1,.06)
 return sh
def banner(s,msg,t=462):
 box(s,35,t,890,36,rgb(232,242,250));tx(s,47,t+5,866,28,msg,14,DBLUE,True)
def pic(s,name,x,y,w,h):
 p=B/name if (B/name).exists() else Path(FIG)/name
 im=Image.open(p); r=min(w/im.width,h/im.height)
 sh=s.Shapes.AddPicture(str(p),0,-1,x+(w-im.width*r)/2,y+(h-im.height*r)/2,im.width*r,im.height*r)
 sh.Line.Visible=-1;sh.Line.ForeColor.RGB=rgb(215,222,228);sh.Line.Weight=.75
 return sh
def panel(s,x,y,w,h,head,body,c=BLUE):
 box(s,x,y,w,h,WHITE,c);tx(s,x+12,y+9,w-24,26,head,16,c,True)
 tx(s,x+12,y+43,w-24,h-48,body,14)
def trip(s,items,y=343,h=105):
 for j,(head,body,c) in enumerate(items):panel(s,35+j*302,y,286,h,head,body,c)
def title(s,msg):
 if s.Shapes.Title is not None:
  sh=s.Shapes.Title;sh.TextFrame.TextRange.Text=msg;sh.Left=35;sh.Top=15;sh.Width=745;sh.Height=45;style(sh.TextFrame.TextRange,28,TXT,False,'方正大标宋简体')
 else:tx(s,35,15,745,45,msg,28,TXT)
def icons(s,n,coords):
 donor=S['icons3'] if n==3 else S['icons4']
 # Native circular pictograms are selected by size and position, without importing text.
 candidates=[]
 for j,(x,y) in enumerate(coords):
  if candidates:
   candidates[j%len(candidates)].Copy();obj=s.Shapes.Paste().Item(1);obj.Left=x;obj.Top=y;obj.Width=42;obj.Height=42
  else:
   # Crop from the native template preview preserves its own circular pictogram.
   tp=B.parent/'template_preview'/('幻灯片19.PNG' if n==3 else '幻灯片20.PNG')
   im=Image.open(tp); ww,hh=im.size
   xx,yy=([(200,187),(483,187),(766,187)] if n==3 else [(376,181),(581,181),(378,393),(584,393)])[j%n]
   im.crop((int((xx-37)/960*ww),int((yy-37)/540*hh),int((xx+37)/960*ww),int((yy+37)/540*hh))).save(B/f'icon{n}_{j}.png')
   pic(s,f'icon{n}_{j}.png',x,y,42,42)
donors={v.SlideID for k,v in S.items() if k in ['icons3','icons4','timeline']}
ss=[s for s in pres.Slides if s.SlideID not in donors]
assert len(ss)==23,len(ss)
ss[5],ss[6]=ss[6],ss[5]
ss[8],ss[9]=ss[9],ss[8]
ss[17],ss[18]=ss[18],ss[17]
for i,s in enumerate(ss,1):s.MoveTo(i)
for i in [4,5,6,8,9,11,12,13,14,15,16,17,18,20,21,22]:wipe(ss[i-1])
print('Wiped content',flush=True)
s=ss[3];title(s,'真实采集场景更需要检验模型依据')
heads=['低空安全需求','射频检测的优势','实测场景的难点']
bodies=['复杂电磁环境下\r及时发现无人机','被动接收通信信号\r可进一步识别机型','模型容易走捷径\r实验室高分≠实测可用']
icons(s,3,[(158,118),(460,118),(762,118)])
for j in range(3):
 tx(s,35+j*302,192,286,32,heads[j],20,ORANGE if j==2 else BLUE,True,2)
 tx(s,47+j*302,252,262,100,bodies[j],18,align=2)
banner(s,'目标：在真实采集数据上建立可信的检测与识别方法',420)
s=ss[4];title(s,'先检测有无无人机，再识别具体型号')
for j,t in enumerate(['原始IQ信号','时频图 +\rIQ手工特征','第一阶段\r有无无人机','第二阶段\r识别型号']):
 b=box(s,35+j*228,96,206,76,DBLUE if j<2 else BLUE);tx(s,39+j*228,109,198,56,t,18,WHITE,True,2)
 if j<3:
  a=s.Shapes.AddShape(33,244+j*228,125,16,18);a.Fill.ForeColor.RGB=ORANGE;a.Line.Visible=0
trip(s,[('第一阶段','各机型漏检 ≤1%\r背景虚警 ≤1%',BLUE),('低信噪比','不加噪、0、−5、−10、−15 dB\r平均准确率 ≥95%',BLUE),('第二阶段','识别具体型号\r同型号、同图传带宽比较',BLUE)],218,164)
banner(s,'按录音分组 · 分来源统计 · 3个随机种子 · 报告95%置信上界',428)
s=ss[5];title(s,'文献对标：从随机高分转向可靠评测')
rows=[['方面','主流做法（代表文献）','存在的不足','本项目做法'],['输入表示','时频图 + CNN\rDroneRFa 2024 / Ozturk 2020','短窗口容易漏掉突发','ResNet18基线\r比较10–50 ms窗长'],['特征融合','统计特征 / 深浅拼接\rEzuma 2020 / Yang、Han 2021','可能携带接收机指纹','先检验来源\r再做嵌入层拼接'],['数据划分','随机切分窗口\rDroneRFa 6:2:2','同录音相邻片段泄漏','按录音分组\r训练/测试零泄漏'],['泛化评测','总体准确率\rWiSig / Geirhos 2020','换接收机99%降至<33%','背景替换与噪声对照\r分接收机统计']]
tb=s.Shapes.AddTable(5,4,35,90,890,316).Table
for j,w in enumerate([100,310,220,260],1):tb.Columns(j).Width=w
for i,row in enumerate(rows,1):
 for j,t in enumerate(row,1):
  sh=tb.Cell(i,j).Shape;sh.TextFrame.TextRange.Text=t;style(sh.TextFrame.TextRange,14 if i>1 else 15,WHITE if i==1 else TXT,i==1)
  sh.Fill.ForeColor.RGB=BLUE if i==1 else (rgb(232,242,250) if j==4 else WHITE)
tx(s,35,432,890,40,'高权威文献34篇（精读10篇）· 融合文献19篇（全文12篇）· 公开WiFi/蓝牙数据集21个',12,DBLUE)
s=ss[7];title(s,'技术路线：前六步完成，输入改进进行中')
flow=['文献调研','公开数据集基线','实测审核预处理','第一阶段检测','捷径排查与修正','时频图+IQ融合','频带截取+\r公开干扰叠加','低信噪比训练','第二阶段识别']
for j,t in enumerate(flow):
 row,col=divmod(j,3);col=2-col if row==1 else col;x=35+col*302;y=95+row*112;c=BLUE if j<6 else ORANGE if j==6 else GRAY
 box(s,x,y,270,77,c);tx(s,x+8,y+15,254,48,t,17,WHITE,True,2)
 if j<8:
  a=s.Shapes.AddShape(36 if j%3==2 else 34 if row==1 else 33,x+125 if j%3==2 else x-27 if row==1 else x+276,y+82 if j%3==2 else y+29,20,22);a.Fill.ForeColor.RGB=rgb(180,200,215);a.Line.Visible=0
tx(s,35,455,890,30,'蓝色：已完成　　橙色：进行中　　灰色：计划',14,DBLUE)
s=ss[8];title(s,'多轮对照实验形成可追溯的工作积累')
for j,(n,l) in enumerate([('约360 GB','处理的射频数据'),('6400+','原始录音文件'),('60+','模型训练（11月）'),('26','实验任务书'),('20','实验报告'),('100+','分析图')]):
 x=35+j*151;box(s,x,95,135,104,LGRAY);tx(s,x,106,135,42,n,25,BLUE,True,2);tx(s,x,155,135,32,l,14,align=2)
line=s.Shapes.AddLine(65,283,895,283);line.Line.ForeColor.RGB=BLUE;line.Line.Weight=2
for j,(m,t) in enumerate([('9月','文献调研\rDroneRFa基线'),('10月','基线诊断\r转实测、全面审核'),('11月上旬','截频切片与训练\r捷径排查与修正'),('11月中旬','特征融合、频带截取\r公开数据调研')]):
 x=35+j*228;box(s,x+98,275,16,16,BLUE);tx(s,x,234,206,35,m,18,DBLUE,True,2);tx(s,x,312,206,85,t,15,align=2)
s=ss[10];title(s,'公开数据基线暴露短窗口漏信号问题')
pic(s,'图A_DroneRFa两步识别_测试集_不加噪.png',35,84,550,376)
panel(s,610,90,315,160,'数据与方法','12种机型 + 背景，按文件划分\r10 ms时频图 + ResNet18\r不加噪85.4%，−15 dB为51.1%\rMatrice 300仅34.8%',BLUE)
panel(s,610,272,315,190,'发现：需比较窗长','Matrice 100每14 ms发1–2 ms突发\r10 ms窗约29%抓不到信号\r\r由错误样本回到信号本身',ORANGE)
s=ss[11];title(s,'难题一：先清理数据，再解释模型结果')
im=Image.open(B/'lines.png');im.crop((0,1000,im.width,1315)).save(B/'lines_detail.png')
pic(s,'lines_detail.png',35,85,890,230)
trip(s,[('质量核验','97重复 / 15全零 / 25前段为零\r核验剔除；按录音分组零泄漏',BLUE),('采样率确认','同步序列匹配推断\r122.88 MS/s，并经确认',BLUE),('本振谱线清理','正弦对消 / 陷波\r所有样本使用同一陷波表',BLUE)],340,119)
s=ss[12];title(s,'难题二：模型把 WiFi 当成背景')
pic(s,'background.png',35,80,890,255)
trip(s,[('问题','第一轮虚警 <1%\r模型到底依据什么？',ORANGE),('对照','同一批样本、相同功率比\r只换白噪声 / 城区背景',BLUE),('发现','2.4 GHz、0 dB：90%降至57%\r城区样本加白噪声反而易检出',DBLUE)],348,107)
tx(s,35,472,890,24,'5.8 GHz未见此现象。',11,GRAY)
s=ss[13];title(s,'难题三：同录音镜像构造接收机背景')
pic(s,'bgA.png',35,80,565,240)
tx(s,617,86,308,30,'共轭后镜像至 fc 另一侧',15,BLUE,True)
line=s.Shapes.AddLine(625,220,915,220);line.Line.ForeColor.RGB=DBLUE;line.Line.EndArrowheadStyle=3
box(s,630,185,105,25,BLUE);box(s,796,185,112,25,DBLUE)
tx(s,617,233,150,45,'空闲段\r2342–2393 MHz',12,BLUE,align=2);tx(s,775,233,150,45,'工作频带\r2405–2455 MHz',12,DBLUE,align=2)
arc=s.Shapes.AddShape(25,680,128,173,65);arc.Fill.Visible=0;arc.Line.ForeColor.RGB=BLUE;arc.Line.Weight=2;arc.Line.EndArrowheadStyle=3
trip(s,[('问题','A仅录无人机，缺少同接收机背景\r模型把干净底噪当作标签',ORANGE),('解决','空闲频段共轭镜像\r检验镜像泄漏，剔除不合格块',BLUE),('结果','无人机召回、城区背景虚警不变',DBLUE)],348,116)
tx(s,655,418,245,35,'100% → 0%',24,DBLUE,True,2)
s=ss[14];title(s,'难题四：WiFi 干扰必须对称叠加')
pic(s,'wifi.png',35,84,890,242)
trip(s,[('La：只叠无人机','A底噪 + WiFi形成新捷径\r城区叠加集虚警7.3%升至24%',ORANGE),('Lb：无人机与背景都叠加','城区叠加集虚警1.9%\r城区无人机召回62%升至81%',BLUE),('结果','采用 Lb 作为2.4 GHz基线\r干扰叠加必须对称',DBLUE)],348,116)
s=ss[15];title(s,'窗长比较：10–50 ms 的覆盖与分辨率取舍')
pic(s,'window.png',35,82,580,385)
panel(s,640,90,285,160,'设计','10/20/30/40/50 ms\r两个频段 × 3种子\r一次只改窗长\rargmax，均值±标准差',BLUE)
panel(s,640,272,285,184,'取舍与结论','窗越长，越能覆盖突发周期\r样本越少、时间分辨率越低\r\r结论：【待补】',ORANGE)
s=ss[16];title(s,'IQ 特征融合尚未带来稳定提升')
pic(s,'fusion_focus.png',35,82,580,325)
panel(s,640,90,285,294,'设计与结果','ResNet18 512维 + 特征64维\r来源检验后做嵌入层拼接\rR / Z / F / F5 × 3种子\r\r2.4 GHz：F虚警2.5%\r三种子改善方向不一致\r5.8 GHz：召回下降5.6–8.7个百分点',BLUE)
banner(s,'手工特征可能携带录音来源信息，必须先检验来源再融合',428)
s=ss[17];title(s,'难题五：频带截取在城区 2.4 GHz 仍受阻')
pic(s,'normal.png',35,107,433,198);pic(s,'merged.png',492,107,433,198)
tx(s,35,78,433,28,'正常：实验室，多频带边界清楚',14,BLUE,True)
tx(s,492,78,433,28,'反例：城区背景，频带被合并',14,ORANGE,True)
outline=box(s,535,148,369,95,WHITE,ORANGE);outline.Fill.Visible=0;outline.Line.Weight=2
tx(s,580,116,325,26,'被合并的宽频带',14,ORANGE,True)
trip(s,[('问题与进展','Mavic 2信号99%在2460–2475 MHz\r实验室、城区5.8 GHz检测正常',ORANGE),('未解决的难点','城区2.4 GHz WiFi与图传交错\r频带可被合并成60–68 MHz',ORANGE),('下一步','按频点同开同关相关性切分\r公开干扰数据选定 NIST',BLUE)],348,116)
tx(s,35,473,890,23,'公开WiFi/蓝牙数据集已调研21个，仅下载需要的录音。',11,GRAY)
s=ss[19];title(s,'创新围绕实测数据的捷径检验展开')
icons(s,4,[(53,104),(508,104),(53,284),(508,284)])
data=[('捷径检验流程','背景替换、噪声对照、分接收机统计'),('不补录构造同接收机背景','同录音空闲频段镜像，虚警100%降至0%'),('对称干扰叠加','城区叠加集虚警：24% vs 1.9%'),('先检验来源再融合','IQ特征来源检验；按信号频带截取进行中')]
for j,(h,t) in enumerate(data):
 x=35+(j%2)*455;y=93+(j//2)*180
 tx(s,x+78,y+18,345,54,h,19,BLUE,True);tx(s,x+18,y+87,405,57,t,15)
tx(s,35,464,890,28,'据目前检索，相关工作较少系统检验模型对来源线索的依赖。【答辩前再核对】',11,GRAY)
s=ss[20];title(s,'当前仍有六项限制需要后续验证')
problems=['背景仅80片：0/80的95%上界3.7%，尚不能验证虚警≤1%','5.8 GHz缺实验室背景，接收机与标签混淆','实验室2.4 GHz仅录到2461 MHz，高频图传缺失','城区2.4 GHz频带切分未解决','IQ特征低信噪比效果待复核','仅一块8 GB显卡，长窗和IQ训练受限']
for j,t in enumerate(problems):
 a=s.Shapes.AddShape(7,42,100+j*61,13,13);a.Fill.ForeColor.RGB=ORANGE;a.Line.Visible=0
 tx(s,70,91+j*61,842,47,t,16)
s=ss[21];title(s,'下一阶段按输入改进、达标与识别推进')
line=s.Shapes.AddLine(62,186,898,186);line.Line.ForeColor.RGB=BLUE;line.Line.Weight=2
plan=[('2026.11—12','公开干扰对称叠加\r城区2.4 GHz频带切分\r按信号频带重训'),('2027.01','第一阶段达标\r漏检、虚警均≤1%\r报告置信上界'),('2027.02—03','低信噪比训练\r平均准确率≥95%'),('2027.03—04','第二阶段型号识别\r泛化测试'),('2027.04—05','补全消融与对照\r论文与结题报告')]
for j,(d,t) in enumerate(plan):
 x=35+j*182;c=ORANGE if j==0 else GRAY
 dot=s.Shapes.AddShape(9,x+73,177,18,18);dot.Fill.ForeColor.RGB=c;dot.Line.Visible=0
 tx(s,x,126,162,38,d,16,c,True,2);panel(s,x,225,162,190,'进行中' if j==0 else '计划',t,c)
tx(s,35,464,890,28,'【结题时间待核对】',11,GRAY)
# Finish notes, remove every unused donor, and normalize inherited footer text.
print('Content overlay complete',flush=True)
set_notes(ss[1],'下面从背景与目标、技术路线、已完成工作和关键难题、创新点与计划四部分汇报。')
for i in [3,7,10,19]:set_notes(ss[i-1],['这一部分介绍研究背景、两阶段目标及文献对标。','这一部分介绍总体路线及已完成的工作量。','这一部分重点汇报实测数据中的五个难题、对照实验和修正结果。','最后介绍创新点、当前局限与下一阶段计划。'][[3,7,10,19].index(i)])
for key in ['icons3','icons4','timeline']:S[key].Delete()
bad=['请在此','这是此模版','单击添加','关键词','学校肇始','芃苇','PengV','Page']
for master in pres.Designs:
 print('Cleaning master',flush=True)
 for container in [master.SlideMaster]+list(master.SlideMaster.CustomLayouts):
  for sh in walk(container.Shapes):
   if sh.HasTextFrame and sh.TextFrame.HasText:
    tr=sh.TextFrame.TextRange;t=tr.Text
    if any(b in t for b in bad):tr.Text='' if 'Page' not in t else t.replace('Page','')
for i,s in enumerate(pres.Slides,1):
 if i in [3,7,10,19]:
  for sh in s.Shapes:
   if sh.HasTextFrame and sh.TextFrame.HasText and sh.TextFrame.TextRange.Text in ['壹','贰','叁','肆']:sh.Height=225
 if i not in [1,2,3,7,10,19,23]:
  for sh in s.Shapes:
   if sh.Type==1:
    try:
     visible=sh.Fill.Visible;color=sh.Fill.ForeColor.RGB;sh.Fill.Solid();sh.Fill.ForeColor.RGB=color;sh.Fill.Visible=visible
    except:pass
 for sh in walk(s.Shapes):
  if sh.HasTextFrame and sh.TextFrame.HasText:
   tr=sh.TextFrame.TextRange
   if any(b in tr.Text for b in bad):tr.Text=''
 # All notes contain actual prose, including native template pages.
 for k in range(1,s.TimeLine.MainSequence.Count+1):
  try:s.TimeLine.MainSequence.Item(1).Delete()
  except:break
 s.SlideShowTransition.EntryEffect=0
print('Saving',OUT,flush=True)
pres.SaveAs(OUT,24)
print('Saved, exporting',flush=True)
for i,s in enumerate(pres.Slides,1):s.Export(str(P/f'{i:02d}.png'),'PNG',1920,1080)
print('overlay and export complete',flush=True)
