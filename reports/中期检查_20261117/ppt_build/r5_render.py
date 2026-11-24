from pathlib import Path
import json
import sys
from PIL import Image, ImageDraw
import win32com.client as win32

B=Path(__file__).resolve().parent
D=B.parent
round_id=int(sys.argv[1])
prefix='r5_round1' if round_id==1 else 'r5'
destination=D/'ppt_preview'
app=win32.DispatchEx('PowerPoint.Application')
deck=app.Presentations.Open(str(B/'r5_work.pptx'),True,False,False)
data=[]
def inspect(sh, is_cell=False):
    result=[]
    if is_cell:
        tr=sh.TextFrame.TextRange
        return [{'name':'table cell','text':tr.Text,'size':float(tr.Font.Size)}]
    if sh.Type==6:
        for cell in sh.GroupItems:
            result.extend(inspect(cell))
    elif not is_cell and sh.HasTable:
        for j in range(1,sh.Table.Rows.Count+1):
            for k in range(1,sh.Table.Columns.Count+1):
                result.extend(inspect(sh.Table.Cell(j,k).Shape, True))
    elif sh.HasTextFrame and sh.TextFrame.HasText:
        rect=[float(v) for v in (sh.Left,sh.Top,sh.Width,sh.Height)]
        tr=sh.TextFrame.TextRange
        bounds=[float(v) for v in (tr.BoundLeft,tr.BoundTop,tr.BoundWidth,tr.BoundHeight)]
        result.append({'name':sh.Name,'text':tr.Text,'rect':rect,'bounds':bounds,'size':float(tr.Font.Size)})
    return result
try:
    assert deck.Slides.Count==19,deck.Slides.Count
    for slide in deck.Slides:
        i=slide.SlideIndex
        slide.Export(str(destination/f'{prefix}_{i:02d}.png'),'PNG',1920,1080)
        text=[]
        for sh in slide.Shapes:
            text.extend(inspect(sh))
        data.append({'page':i,'text':text})
        print(f'Exported {i:02d}',flush=True)
finally:
    deck.Close()
(B/f'r5_round{round_id}_render.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
sheet=Image.new('RGB',(1920,1450),'#e2eaf0')
d=ImageDraw.Draw(sheet)
for i in range(19):
    with Image.open(destination/f'{prefix}_{i+1:02d}.png') as im:
        assert im.size==(1920,1080)
        x=i%4*480
        y=i//4*290
        sheet.paste(im.resize((480,270),Image.Resampling.LANCZOS),(x,y))
        d.text((x+6,y+272),str(i+1),fill='black')
sheet.save((B/'r5_round1_contact.png') if round_id==1 else (D/'contact_r5.png'))



