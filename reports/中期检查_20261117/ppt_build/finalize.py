from pathlib import Path
import os,shutil
import win32com.client as w
B=Path(__file__).resolve().parent
src=max(B.glob('build_*.pptx'),key=lambda p:p.stat().st_mtime)
a=w.DispatchEx('PowerPoint.Application');p=a.Presentations.Open(str(src),False,False,False);a.DisplayAlerts=1
try:
 s=p.Slides(12)
 for i in range(s.Shapes.Count,0,-1):
  sh=s.Shapes(i)
  if sh.Type==13 and sh.Top>75:sh.Delete()
 sh=s.Shapes.AddPicture(str(B/'lines_candidate.png'),0,-1,35,119,890,890*315/1820)
 sh.Line.Visible=-1;sh.Line.Weight=.75;sh.Line.ForeColor.RGB=0xE4DED7
 # The key percentages are editable emphasis, separate from supporting detail.
 s=p.Slides(15)
 for i in range(s.Shapes.Count,0,-1):
  sh=s.Shapes(i)
  if sh.HasTextFrame and sh.TextFrame.HasText and sh.TextFrame.TextRange.Text in ['24%','1.9%','7.3% → 24%']:sh.Delete()
 for sh in s.Shapes:
  if sh.HasTextFrame and sh.TextFrame.HasText:
   t=sh.TextFrame.TextRange.Text
   if 'A底噪 + WiFi形成新捷径' in t:sh.TextFrame.TextRange.Text='A底噪 + WiFi形成新捷径';sh.Height=26
   if '城区叠加集虚警1.9%' in t:sh.TextFrame.TextRange.Text='城区无人机召回62%升至81%';sh.Top=393;sh.Height=27
 for x,y,t,c in [(57,428,'7.3% → 24%',0x0AA3F0),(660-302,428,'1.9%',0x8C5A00)]:
  sh=s.Shapes.AddTextbox(1,x,y,232,29);tf=sh.TextFrame;tf.MarginTop=0;tf.TextRange.Text=t;tf.TextRange.Font.Name='微软雅黑';tf.TextRange.Font.NameFarEast='微软雅黑';tf.TextRange.Font.Size=20;tf.TextRange.Font.Bold=-1;tf.TextRange.Font.Color.RGB=c
 for sh in p.Slides(18).Shapes:
  if sh.Type==1 and abs(sh.Top-148)<1 and abs(sh.Width-369)<1:sh.Fill.Visible=0
 for s in [p.Slides(1),p.Slides(23)]:
  for sh in s.Shapes:
   if sh.HasTextFrame and sh.TextFrame.HasText:
    t=sh.TextFrame.TextRange.Text
    if 'XXX' in t:sh.TextFrame.TextRange.Text=t.replace('XXX','xxx')
 out=B/f'build_final_{os.getpid()}.pptx';print('saving final',out,flush=True);p.SaveAs(str(out),24);print('saved final',flush=True)
finally:
 p.Saved=True;p.Close()
 try:a.Quit()
 except:pass
