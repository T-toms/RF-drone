from pathlib import Path
B=Path(__file__).resolve().parent
s=(B/'r3_finalize.py').read_text(encoding='utf-8-sig');prefix=s.split("app=w.DispatchEx")[0];tail=s[s.index(" out=B/'r3_round2.pptx'"):s.index('finally:\n try:p.Close()')]
tail=tail.replace("r3_round2.pptx","r3_final.pptx").replace('r3_round2_{i:02d}','r3_round3_{i:02d}').replace("'r3_audit.json'","'r3_final_audit.json'")
body='''app=w.DispatchEx('PowerPoint.Application');p=app.Presentations.Open(str(B/'r3_round2.pptx'),False,False,False);app.DisplayAlerts=1
try:
 s=p.Slides(5)
 for sh in list(s.Shapes):
  if sh.Type==9 and abs(sh.Left-865)<1 and abs(sh.Top-284)<1:sh.Delete()
 line(s,908,258,920,258,False);line(s,920,258,920,321,False);line(s,920,321,567,321,False)
 # Make actual text frames enclose the rendered lines after WPS font reflow.
 for ss in p.Slides:
  for sh in list(ss.Shapes):
   if sh.Type==17 and sh.HasTextFrame and sh.TextFrame.HasText:
    tr=sh.TextFrame.TextRange
    if tr.BoundWidth>sh.Width+3 and sh.Top>70 and sh.Top<500 and tr.Font.Size>=12:
     tr.Font.Size=12
    if tr.BoundHeight>sh.Height-2:sh.Height=tr.BoundHeight+4
 # Retain figure captions and dimensions at 10 pt; larger numbers at 22 pt.
'''
(B/'r3_repair.py').write_text(prefix+body+tail+'''finally:
 try:p.Close()
 except:pass
 try:app.Quit()
 except:pass
''',encoding='utf-8')
