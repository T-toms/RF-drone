from pathlib import Path
import win32com.client as w,json,re,hashlib
B=Path(__file__).resolve().parent;D=B.parent
app=w.DispatchEx('Word.Application');app.Visible=False;app.DisplayAlerts=0
try:
 doc=app.Documents.Open(str(D/'中期检查表_已填.doc'),False,True);t=doc.Tables(1)
 data={'pages':doc.ComputeStatistics(2),'words':doc.ComputeStatistics(0),'chars':doc.ComputeStatistics(3),'cells':{},'schedule':[]}
 for r in [11,12]:data['cells'][str(r)]=t.Cell(r,1).Range.Text
 for r in range(15,22):data['schedule'].append([t.Cell(r,c).Range.Text.strip('\r\x07') for c in [1,2]])
 data['section4']=t.Cell(22,1).Range.Text
 data['empty_paragraphs']={k:len(re.findall(r'\r[ \t　]*\r',v)) for k,v in data['cells'].items()}
 md=(D/'中期检查表_一二三项_原稿.md').read_text(encoding='utf-8');expected=[]
 for line in md.splitlines():
  if line.startswith('|') and '---' not in line and '起止时间' not in line:expected.append([x.strip() for x in line.strip('|').split('|')])
 data['schedule_match']=expected==data['schedule']
 for key,sec in [('11','一'),('12','二')]:
  block=re.search(r'^## '+sec+r'\s*$(.*?)(?=^## |\Z)',md,re.M|re.S)[1]
  paras=[re.sub(r'^### |\*\*','',x).strip() for x in block.splitlines() if x.strip()]
  data['cells'][key+'_complete']=all(x in data['cells'][key] for x in paras)
 (B/'r3_form_readback.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 print({k:v for k,v in data.items() if k not in ['cells','schedule']},flush=True)
 doc.Close(False)
finally:
 try:app.Quit()
 except:pass
