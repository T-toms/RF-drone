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
app=w.DispatchEx('PowerPoint.Application');p=app.Presentations.Open(str(D/'中期答辩_模板稿.pptx'),False,False,False)
try:
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
  s.Export(str(P/f'r3_delivery_{i:02d}.png'),'PNG',1920,1080);s.Export(str(P/f'r3_{i:02d}.png'),'PNG',1920,1080)
 (B/'r3_delivery_audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');p.Close()
 try:app.Quit()
 except:pass
 print('Final reopened',len(data),'pages',flush=True)
finally:
 try:p.Close()
 except:pass
 try:app.Quit()
 except:pass
