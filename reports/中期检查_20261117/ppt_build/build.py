from pathlib import Path
import os,json
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
D=Path(__file__).resolve().parent
src=(D.parent/'build_ppt.py').read_text(encoding='utf-8-sig')
src=src.replace("D = os.path.dirname(os.path.abspath(__file__))", "D = str(Path(__file__).resolve().parent.parent)")
src=src.replace("pres = app.Presentations.Open(TPL, False, False, False)","print('COM ready',flush=True)\npres = app.Presentations.Open(TPL, False, False, False)\nprint('Template opened',flush=True)")
src=src.replace('pres.SaveAs(OUT)',"app.DisplayAlerts = 1\nprint('Working in memory; save only after pruning',flush=True)")
src=src.replace("OUT = os.path.join(D, '中期答辩_模板稿.pptx')","OUT = str(Path(__file__).resolve().parent/('build_'+str(os.getpid())+'.pptx'))")
src=src.replace("def new_slide(title, notes=''):","def new_slide(title, notes=''):\n    print('Creating',title,flush=True)")
src=src.replace("keep = {1: 'cover', 5: 'toc', 6: 'trans', 44: 'end'}","keep = {1: 'cover', 5: 'toc', 6: 'trans', 19:'icons3',20:'icons4',30:'timeline',44: 'end'}")
src=src.replace("S = {name: pres.Slides.FindBySlideID(sid) for name, sid in ids.items()}","S = {name: pres.Slides(j) for j,name in enumerate(keep.values(),1)}")
src=src.replace("pres.Save()\nprint('slides:'", "exec((Path(__file__).resolve().parent/'overlay.py').read_text(encoding='utf-8'), globals())\npres.Save()\nprint('slides:'")
try:
 (D/'build_generated.py').write_text(src,encoding='utf-8')
 exec(compile(src,str(D/'build_generated.py'),'exec'))
finally:
 try:pres.Saved=True;pres.Close()
 except:pass
 try:app.Quit()
 except:pass
