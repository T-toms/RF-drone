from pathlib import Path
import os,json,re,csv,zipfile,hashlib,shutil
from datetime import datetime
from PIL import Image,ImageDraw
import win32com.client as win32
B=Path(__file__).resolve().parent;D=B.parent;P=D/'ppt_preview'
src=max(B.glob('build_*.pptx'),key=lambda p:p.stat().st_mtime)
out=D/'中期答辩_模板稿.pptx';shutil.copy2(src,out)
a=win32.DispatchEx('PowerPoint.Application');p=a.Presentations.Open(str(out),True,False,False)
def walk(shapes):
 for sh in shapes:
  if sh.Type==6:yield from walk(sh.GroupItems)
  elif sh.HasTable:
   for i in range(1,sh.Table.Rows.Count+1):
    for j in range(1,sh.Table.Columns.Count+1):yield sh.Table.Cell(i,j).Shape
  else:yield sh
bad=['请在此','这是此模版','单击添加','关键词','学校肇始','芃苇','PengV','Page']
stats=[];alltexts=[];overflow=[];notes=[]
try:
 for i,s in enumerate(p.Slides,1):
  ts=[];sizes=[];bodychars=0
  for sh in walk(s.Shapes):
   if sh.HasTextFrame and sh.TextFrame.HasText:
    tr=sh.TextFrame.TextRange;t=tr.Text;ts.append(t)
    for j in range(1,tr.Length+1):
     try:sizes.append(float(tr.Characters(j,1).Font.Size))
     except:pass
    try:top=sh.Top
    except:top=90
    if top>=75 and top<500:bodychars+=len(re.sub(r'\s','',t))
    try:
     if tr.BoundHeight>sh.Height+3 or tr.BoundWidth>sh.Width+3:overflow.append([i,sh.Name,t,tr.BoundHeight,sh.Height,tr.BoundWidth,sh.Width])
    except:pass
  note='\n'.join(sh.TextFrame.TextRange.Text for sh in walk(s.NotesPage.Shapes) if sh.HasTextFrame and sh.TextFrame.HasText)
  notes.append(note)
  stats.append({'page':i,'characters':sum(len(re.sub(r'\s','',t)) for t in ts),'body_characters':bodychars,'min_font':min(sizes) if sizes else None,'texts':ts})
  alltexts+=ts
  s.Export(str(P/f'{i:02d}.png'),'PNG',1920,1080)
 for des in p.Designs:
  for container in [des.SlideMaster]+list(des.SlideMaster.CustomLayouts):
   alltexts += [sh.TextFrame.TextRange.Text for sh in walk(container.Shapes) if sh.HasTextFrame and sh.TextFrame.HasText]
 print('reopened',p.Slides.Count,'slides',flush=True)
finally:
 p.Close()
 try:a.Quit()
 except:pass
hits={b:sum(t.count(b) for t in alltexts) for b in bad}
(B/'audit.json').write_text(json.dumps({'slides':stats,'residual':hits,'overflow':overflow,'notes':notes},ensure_ascii=False,indent=2),encoding='utf-8')
# Every slide is opened as a separate PNG and prepared for readable per-page review.
for i in range(1,24):
 im=Image.open(P/f'{i:02d}.png');im.resize((960,540)).save(P/f'review_{i:02d}.png')
contact=Image.new('RGB',(1920,1740),'#dfe5ea');dr=ImageDraw.Draw(contact)
for i in range(1,24):
 im=Image.open(P/f'{i:02d}.png').resize((480,270));x=(i-1)%4*480;y=(i-1)//4*290;contact.paste(im,(x,y));dr.text((x+8,y+270),str(i),fill='black')
contact.save(P/'contact.png');contact.save(P/'contact_round2.png')
print('residual',hits,'overflows',overflow,flush=True)
