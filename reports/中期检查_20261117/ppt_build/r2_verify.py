from pathlib import Path
import win32com.client as w
import json,os,shutil,re
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;D=B.parent;P=D/'ppt_preview';out=D/'中期答辩_模板稿.pptx'
data=json.loads((B/'r2_audit_round4.json').read_text(encoding='utf-8'))
a=w.DispatchEx('PowerPoint.Application');p=a.Presentations.Open(str(out),False,False,False);a.DisplayAlerts=1
def walk(shapes):
 for sh in shapes:
  if sh.Type==6:yield from walk(sh.GroupItems)
  elif sh.HasTable:
   for i in range(1,sh.Table.Rows.Count+1):
    for j in range(1,sh.Table.Columns.Count+1):yield sh.Table.Cell(i,j).Shape
  else:yield sh
try:
 changed=0
 for sh in p.Slides(17).Shapes:
  if sh.HasTextFrame and sh.TextFrame.HasText and sh.TextFrame.TextRange.Text.startswith('图源：Gao'):
   sh.TextFrame.TextRange.Text='图源：Gao等，JSAC 2026稿，Fig.2局部，p.5。\rC为拼接，再送入网络头（非加法⊕）。'
 for s in p.Slides:
  for sh in walk(s.Shapes):
   if sh.HasTextFrame and sh.TextFrame.HasText:
    try:
     fs=float(sh.TextFrame.TextRange.Font.Size)
     if fs>0 and fs<10:sh.TextFrame.TextRange.Font.Size=10;changed+=1
    except:pass
 print('Caption minimum normalized',changed,flush=True)
 final=B/f'r2_final_{os.getpid()}.pptx';p.SaveAs(str(final),24);p.Close()
 try:a.Quit()
 except:pass
 shutil.copy2(final,out)
 import subprocess, sys, time
 time.sleep(5)
 subprocess.run([sys.executable, '-I', str(B/'r2_reopen.py')], check=True)
finally:
 try:p.Close()
 except:pass
 try:a.Quit()
 except:pass
