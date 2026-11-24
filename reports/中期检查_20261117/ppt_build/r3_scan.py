from pathlib import Path
import zipfile,xml.etree.ElementTree as E
B=Path(__file__).resolve().parent
with zipfile.ZipFile(B/'r3_round1.pptx') as z:
 for n in z.namelist():
  if n.endswith('.xml') and n.startswith(('ppt/slides/','ppt/slideLayouts/','ppt/slideMasters/')):
   root=E.fromstring(z.read(n));ts=[t.text for t in root.iter() if t.tag.endswith('}t') and t.text]
   if any('Page' in t for t in ts):print(n,[t for t in ts if 'Page' in t])
