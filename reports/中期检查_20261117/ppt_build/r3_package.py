from pathlib import Path
import zipfile,xml.etree.ElementTree as E,shutil
B=Path(__file__).resolve().parent;D=B.parent;ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for k,v in ns.items():E.register_namespace(k,v)
with zipfile.ZipFile(B/'r3_final.pptx') as z:
 with zipfile.ZipFile(B/'r3_verified.pptx','w',zipfile.ZIP_DEFLATED) as out:
  for n in z.namelist():
   data=z.read(n)
   if n in [f'ppt/slides/slide{i}.xml' for i in [5,7,9,12,14,16]]:
    root=E.fromstring(data);page=int(n.split('slide')[-1].split('.')[0])
    for sp in root.findall('.//p:sp',ns):
     text=''.join(x.text or '' for x in sp.findall('.//a:t',ns));xf=sp.find('p:spPr/a:xfrm',ns)
     if xf is None:continue
     off=xf.find('a:off',ns);ext=xf.find('a:ext',ns);y=int(off.get('y'))/12700
     if page==5 and abs(y-232)<.1 and not text:ext.set('cy',str(65*12700))
     if page==7 and text.startswith('14 ms /'):off.set('y',str(290*12700))
     if page==7 and text.startswith('周期性漏检；'):
      off.set('y',str(487*12700));ext.set('cx',str(890*12700));ext.set('cy',str(20*12700));ts=sp.findall('.//a:t',ns);ts[0].text='每个结论都要配一个“改变无关线索、看结果是否改变”的对照。'
      for t in ts[1:]:t.text=''
     if page==9 and text.startswith('90%  vs'):off.set('y',str(290*12700))
     if page==9 and text.startswith('同一信号，只换背景'):off.set('y',str(325*12700))
     if page==12 and text.startswith('区分能力：'):off.set('y',str(463*12700))
     if page==14 and text.startswith('失败：'):
      ts=sp.findall('.//a:t',ns);ts[0].text='失败：多信号合并的宽带反例（并非正文城区录音）。'
      for t in ts[1:]:t.text=''
     if page==16 and not text:
      pr=sp.find('p:spPr',ns)
      for x in list(pr):
       if x.tag.split('}')[-1] in ['gradFill','solidFill','noFill','pattFill']:pr.remove(x)
      fill=E.SubElement(pr,'{'+ns['a']+'}solidFill');E.SubElement(fill,'{'+ns['a']+'}srgbClr',{'val':'E8F2FA'})
    data=E.tostring(root,encoding='utf-8',xml_declaration=True)
   out.writestr(n,data)
shutil.copy2(B/'r3_verified.pptx',D/'中期答辩_模板稿.pptx')
# Read and export the actual deliverable; no save or modification via COM after packaging.
s=(B/'r3_repair.py').read_text(encoding='utf-8-sig');prefix=s.split("app=w.DispatchEx")[0];audit=s[s.index(' data=[]'):s.index('finally:\n try:p.Close()')];audit=audit.replace("'r3_final_audit.json'","'r3_delivery_audit.json'").replace('r3_round3_{i:02d}','r3_delivery_{i:02d}')
audit=audit[:audit.index(' shutil.copy2')]+" print('Final reopened',len(data),'pages',flush=True)\n"
code=prefix+"app=w.DispatchEx('PowerPoint.Application');p=app.Presentations.Open(str(D/'中期答辩_模板稿.pptx'),False,False,False)\ntry:\n"+audit+"finally:\n try:p.Close()\n except:pass\n try:app.Quit()\n except:pass\n"
(B/'r3_reopen.py').write_text(code,encoding='utf-8')
