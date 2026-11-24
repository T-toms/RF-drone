from pathlib import Path
import win32com.client as w
import zipfile,xml.etree.ElementTree as E,copy
B=Path(__file__).resolve().parent;D=B.parent
app=w.DispatchEx('Word.Application');app.Visible=False;app.DisplayAlerts=0
print('Open original',flush=True)
doc=app.Documents.Open(str(D/'中期检查表_原件.doc'),False,True);doc.SaveAs2(str(B/'r3_form_source.docx'),12);doc.Close(False)
try:app.Quit()
except:pass
print('Converted',flush=True)
src=B/'r3_form_source.docx';out=B/'r3_form_expanded.docx'
with zipfile.ZipFile(src) as z:
 root=E.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};tbl=root.find('.//w:tbl',ns);trs=tbl.findall('w:tr',ns);print('Rows',len(trs),flush=True)
 for i,tr in enumerate(trs):print(i+1,''.join(tr.itertext())[:100],flush=True)
 for i in range(2):tbl.insert(list(tbl).index(trs[18]),copy.deepcopy(trs[18]))
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as zz:
  for n in z.namelist():zz.writestr(n,E.tostring(root,encoding='utf-8',xml_declaration=True) if n=='word/document.xml' else z.read(n))
s=(D/'fill_form.py').read_text(encoding='utf-8-sig').replace("D = os.path.dirname(os.path.abspath(__file__))","D = str(B.parent)")
s=s.replace("SRC_DOC = os.path.join(D, '中期检查表_原件.doc')","SRC_DOC = str(B/'r3_form_expanded.docx')").replace('n_slots = 5','n_slots = 7')
s=s.replace('    doc.Save()','    print("Saving filled form",flush=True)\n    doc.Save()')
exec(compile(s,str(D/'fill_form.py'),'exec'),globals())

