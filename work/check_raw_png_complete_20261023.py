from pathlib import Path
from PIL import Image
import io,json
O=Path(__file__).parent/'audit_drone_20261023'
rows=[]
for p in Path('G:/drone/raw').rglob('*.png'):
 row={'path':str(p)}
 try:
  data=p.read_bytes()
  with Image.open(io.BytesIO(data)) as im:im.verify()
  with Image.open(io.BytesIO(data)) as im:
   im.load();row.update(size=im.size,mode=im.mode,verified=True)
 except Exception as e:row.update(verified=False,error=str(e))
 rows.append(row)
(O/'png_full_decode.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'files':len(rows),'passed':sum(r['verified'] for r in rows),'failed':[r for r in rows if not r['verified']]},ensure_ascii=False))
