from pathlib import Path
import win32com.client as w,json,re,shutil
from PIL import Image
B=Path(__file__).resolve().parent;D=B.parent;P=D/'ppt_preview';A=B/'r3_assets'
exec((B/'r3_notes.py').read_text(encoding='utf-8-sig'),globals())
def rgb(r,g,b):return r+256*g+65536*b
BLUE=rgb(0,121,186);GRAY=rgb(120,120,120);TXT=rgb(64,64,64);WHITE=rgb(255,255,255)
def style(tr,size=12,c=TXT,b=False):
 tr.Font.NameFarEast='微软雅黑';tr.Font.Name='Arial';tr.Font.Size=size;tr.Font.Color.RGB=c;tr.Font.Bold=-1 if b else 0

def tx(s,x,y,ww,hh,t,size=12,c=TXT,b=False):
 sh=s.Shapes.AddTextbox(1,x,y,ww,hh);tf=sh.TextFrame;tf.MarginLeft=tf.MarginRight=1;tf.MarginTop=tf.MarginBottom=1;tf.WordWrap=-1;tf.AutoSize=0;tf.TextRange.Text=t;style(tf.TextRange,size,c,b);tf.TextRange.ParagraphFormat.SpaceAfter=0;return sh

def line(s,x,y,xx,yy,arrow=True):
 sh=s.Shapes.AddLine(x,y,xx,yy);sh.Line.ForeColor.RGB=TXT;sh.Line.Weight=.85
 if arrow:sh.Line.EndArrowheadStyle=3

def replacepic(s,sh,path):
 x,y,ww,hh=sh.Left,sh.Top,sh.Width,sh.Height;sh.Delete();im=Image.open(path);r=min(ww/im.width,hh/im.height);s.Shapes.AddPicture(str(path),0,-1,x+(ww-im.width*r)/2,y+(hh-im.height*r)/2,im.width*r,im.height*r)

def walk(shapes):
 for sh in shapes:
  if sh.Type==6:yield from walk(sh.GroupItems)
  elif sh.HasTable:
   for r in range(1,sh.Table.Rows.Count+1):
    for c in range(1,sh.Table.Columns.Count+1):yield sh.Table.Cell(r,c).Shape
  else:yield sh
app=w.DispatchEx('PowerPoint.Application');p=app.Presentations.Open(str(B/'r3_round1.pptx'),False,False,False);app.DisplayAlerts=1
try:
 for i,s in enumerate(p.Slides,1):
  for sh in list(s.Shapes):
   if sh.HasTextFrame and sh.TextFrame.HasText:
    t=sh.TextFrame.TextRange.Text
    if 'Page' in t:sh.TextFrame.TextRange.Text=''
   if i in [3,4,5] and sh.Type==1:
    try:
     if sh.Line.DashStyle==4:sh.Fill.Visible=-1;sh.Fill.Solid();sh.Fill.ForeColor.RGB=WHITE;sh.ZOrder(1)
    except:pass
  s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text=NOTES[i-1]
 s=p.Slides(3)
 for sh in s.Shapes:
  if sh.HasTextFrame and sh.TextFrame.HasText and sh.TextFrame.TextRange.Text.startswith('难点：'):sh.TextFrame.TextRange.Font.Color.RGB=rgb(240,163,10)
 s=p.Slides(4)
 # Add two cropped, label-free input examples in the workflow's spare bottom band.
 src=Image.open(B.parents[2]/'work/fusionB_B2_20261112/fig0_input_examples_2.4GHz.png')
 for box,name,x,label in [((100,73,625,453),'input_drone.png',485,'无人机'),((1385,73,1910,453),'input_background.png',690,'背景')]:
  src.crop(box).save(A/name);s.Shapes.AddPicture(str(A/name),0,-1,x,192,44,29);tx(s,x+48,194,130,20,label+'输入示例',10,GRAY)
 s=p.Slides(5)
 for sh in s.Shapes:
  if sh.HasTextFrame and sh.TextFrame.HasText:
   t=sh.TextFrame.TextRange.Text
   if sh.Top>230 and sh.Top<240 and ('BN＋ReLU' in t or '2个残差块' in t):sh.Height=65
   if sh.Top>285 and sh.Top<295:sh.Top=302
  if sh.Type==1:
   try:
    if abs(sh.Top-199)<1 and sh.Width>800:sh.Height=130
   except:pass
 # Complete both branches into fusion, using the margins between grouped boxes.
 line(s,865,284,865,321,False);line(s,865,321,567,321,False);line(s,567,321,567,353)
 line(s,460,369,489,369,False);line(s,489,369,500,369)
 # Caption below image was an editing instruction; replace with its scientific meaning.
 for page in [9,10,11,12]:
  s=p.Slides(page)
  for sh in list(s.Shapes):
   if sh.Type!=13:continue
   if page==9 and sh.Left>480:replacepic(s,sh,A/'background_clean.png')
   if page==10 and sh.Top>250:replacepic(s,sh,A/'background_fa_clean.png')
   if page==11 and sh.Top>230:replacepic(s,sh,A/'wifi_focus.png')
   if page==12:replacepic(s,sh,A/'mirror_clean.png')
 s=p.Slides(13)
 for sh in s.Shapes:
  if sh.HasTextFrame and sh.TextFrame.HasText and sh.TextFrame.TextRange.Text.startswith('橙箭头：'):
   sh.TextFrame.TextRange.Text='5.8 GHz：94.9% → 86.2%，3次训练方向一致。\r柱与误差棒：3次独立训练的均值、标准差。'
 for i,num,lab in [(7,'14 ms / 约29%空窗','周期性漏检；100%→0–4%揭示频率位置依赖'),(9,'90%  vs  57%','同一信号，只换背景'),(10,'100% → 0%','补同接收机背景后误报率'),(11,'24%  vs  1.9%','只给无人机叠加 / 对称叠加'),(13,'94.9% → 86.2%','特征全置零 → 5个特征，城区检出下降')]:
  s=p.Slides(i);yy=308 if i in [7,9] else 420;tx(s,35,yy,440,30,num,22,BLUE,True);tx(s,35,yy+32,440,22,lab,12,GRAY)
 # Three content blocks hold the five remaining issues, avoiding long full-width lines.
 s=p.Slides(16);items=[]
 for sh in list(s.Shapes):
  if sh.HasTextFrame and sh.TextFrame.HasText and re.match(r'^[1-5]　',sh.TextFrame.TextRange.Text):items.append(sh.TextFrame.TextRange.Text);sh.Delete()
 for j,(x,y,ww,hh,ids) in enumerate([(35,100,435,192,[0,1]),(490,100,435,192,[2,3]),(35,315,890,135,[4])]):
  rr=s.Shapes.AddShape(1,x,y,ww,hh);rr.Fill.ForeColor.RGB=rgb(232,242,250);rr.Line.Visible=0;tx(s,x+12,y+12,ww-24,hh-24,'\r\r'.join(items[k] for k in ids),13)
 # Explain the unit interval once in the feature-selection page.
 tx(p.Slides(12),628,470,297,15,'区分能力：0.5无法区分，1完全区分',10,GRAY)
 out=B/'r3_round2.pptx';print('Saving round2',flush=True);p.SaveAs(str(out),24);print('Saved; auditing',flush=True)
 data=[]
 for i,s in enumerate(p.Slides,1):
  texts=[];fonts=[];overflow=[]
  for sh in walk(s.Shapes):
   if sh.HasTextFrame and sh.TextFrame.HasText:
    tr=sh.TextFrame.TextRange;texts.append(tr.Text)
    for j in range(1,tr.Runs().Count+1):
     fs=float(tr.Runs(j).Font.Size)
     if fs>0:fonts.append(fs)
    if tr.BoundHeight>sh.Height+3 or tr.BoundWidth>sh.Width+3:overflow.append({'text':tr.Text,'height':sh.Height,'bound_h':tr.BoundHeight,'width':sh.Width,'bound_w':tr.BoundWidth})
  note=s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text
  data.append({'page':i,'chars':sum(len(re.sub(r'\s','',t)) for t in texts),'min_font':min(fonts),'note_chars':len(re.sub(r'\s','',note)),'texts':texts,'notes':note,'overflow':overflow})
  s.Export(str(P/f'r3_round2_{i:02d}.png'),'PNG',1920,1080);s.Export(str(P/f'r3_{i:02d}.png'),'PNG',1920,1080)
 (B/'r3_audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');p.Close()
 try:app.Quit()
 except:pass
 shutil.copy2(out,D/'中期答辩_模板稿.pptx');print('Done',[(d['page'],len(d['overflow'])) for d in data if d['overflow']],flush=True)
finally:
 try:p.Close()
 except:pass
 try:app.Quit()
 except:pass
