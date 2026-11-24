from pathlib import Path
import zipfile,xml.etree.ElementTree as E,shutil
B=Path(__file__).resolve().parent;D=B.parent
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for k,v in ns.items():E.register_namespace(k,v)
adds={10:'背景样本仍按录音分组，防止同一段底噪同时进入训练与测试。',11:'这样两类样本都能见到同一种环境干扰，模型就不能仅凭有无WiFi区分类别；其余训练设置保持一致。',13:'后续先扩充同接收机背景，再重新评估这些特征。'}
with zipfile.ZipFile(D/'中期答辩_模板稿.pptx') as z:
 with zipfile.ZipFile(B/'r3_dense.pptx','w',zipfile.ZIP_DEFLATED) as out:
  for n in z.namelist():
   data=z.read(n)
   for i,extra in adds.items():
    if n==f'ppt/slides/slide{i}.xml':
     root=E.fromstring(data)
     for sp in root.findall('.//p:sp',ns):
      ts=sp.findall('.//a:t',ns);t=''.join(x.text or '' for x in ts)
      if t.startswith('⑤ 结果与留下的问题'):
       ts[-1].text=(ts[-1].text or '')+extra;ext=sp.find('p:spPr/a:xfrm/a:ext',ns);ext.set('cy',str(int(ext.get('cy'))+45*12700))
     data=E.tostring(root,encoding='utf-8',xml_declaration=True)
   out.writestr(n,data)
shutil.copy2(B/'r3_dense.pptx',D/'中期答辩_模板稿.pptx')
