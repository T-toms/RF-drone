from pathlib import Path
import re,json
B=Path(__file__).resolve().parent;D=B.parent
s=(D/'中期检查表_一二三项_原稿.md').read_text(encoding='utf-8')
for m in re.finditer(r'\*\*([1-5]\..*?)\*\*(.*?)(?=\n\*\*|\n###)',s,re.S):print(m[1],len(re.sub(r'\s','',m[2])))
a=json.loads((B/'r3_form_readback.json').read_text(encoding='utf-8'));print(a['cells'].get('11_complete'),a['cells'].get('12_complete'))
