from pathlib import Path
import win32com.client as w
import json,re
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;D=B.parent;P=D/'ppt_preview';out=D/'中期答辩_模板稿.pptx'
data=json.loads((B/'r2_audit_round4.json').read_text(encoding='utf-8'))
a=w.DispatchEx('PowerPoint.Application');p=a.Presentations.Open(str(out),True,False,False)
def walk(shapes):
 for sh in shapes:
  if sh.Type==6:yield from walk(sh.GroupItems)
  elif sh.HasTable:
   for i in range(1,sh.Table.Rows.Count+1):
    for j in range(1,sh.Table.Columns.Count+1):yield sh.Table.Cell(i,j).Shape
  else:yield sh
try:
 assert p.Slides.Count==24
 notes=[];alltext=[];overflows=[]
 for i,s in enumerate(p.Slides,1):
  ts=[];sizes=[];ov=[]
  for sh in walk(s.Shapes):
   if sh.HasTextFrame and sh.TextFrame.HasText:
    tr=sh.TextFrame.TextRange;ts.append(tr.Text)
    try:sizes.append(float(tr.Font.Size))
    except:pass
    try:
     if tr.BoundHeight>sh.Height+3 or tr.BoundWidth>sh.Width+3:ov.append([sh.Name,tr.Text,tr.BoundHeight,sh.Height,tr.BoundWidth,sh.Width])
    except:pass
  nr=[sh.TextFrame.TextRange.Text for sh in walk(s.NotesPage.Shapes) if sh.HasTextFrame and sh.TextFrame.HasText]
  notes.append('\n'.join(nr));alltext+=ts;overflows += [[i]+x for x in ov]
  data['slides'][i-1].update(texts=ts,chars=sum(len(re.sub(r'\s','',t)) for t in ts),min_font=min(v for v in sizes if v>0),overflow=ov)
  s.Export(str(P/f'r2_{i:02d}.png'),'PNG',1920,1080)
 for des in p.Designs:
  for c in [des.SlideMaster]+list(des.SlideMaster.CustomLayouts):
   alltext += [sh.TextFrame.TextRange.Text for sh in walk(c.Shapes) if sh.HasTextFrame and sh.TextFrame.HasText]
 data['notes']=notes;data['reopened_slides']=p.Slides.Count;data['final_build']=str(max(B.glob('r2_final_*.pptx'),key=lambda x:x.stat().st_mtime))
 data['residual']={b:sum(t.count(b) for t in alltext) for b in ['请在此','这是此模版','单击添加','关键词','学校肇始','芃苇','PengV','Page']}
 (B/'r2_final_audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 assert not any(data['residual'].values());assert not overflows,overflows;assert all(len(n)>15 for n in notes)
 print('Reopened exact delivery: 24 slides, zero overflow/residual, 24 notes',flush=True)
finally:
 p.Close()
 try:a.Quit()
 except:pass
im=Image.new('RGB',(1920,1740),'#dfe5ea');dr=ImageDraw.Draw(im)
for i in range(1,25):
 page=Image.open(P/f'r2_{i:02d}.png');page.resize((960,540)).save(P/f'r2_review_{i:02d}.png');x=(i-1)%4*480;y=(i-1)//4*290;im.paste(page.resize((480,270)),(x,y));dr.text((x+8,y+270),str(i),fill='black')
im.save(P/'contact_r2.png')
