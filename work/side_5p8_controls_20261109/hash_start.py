from pathlib import Path
import hashlib,json,time
P=Path('C:/Users/Administrator/Desktop/数据集处理'); O=P/'work/side_5p8_controls_20261109'
t=time.time(); d={}
for name in ['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109','train_stage1_L20_bgA_20261109']:
 for p in sorted((P/'work'/name).rglob('*')):
  if p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   d[str(p.relative_to(P))]=h.hexdigest()
(O/'hash_before.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print(len(d),time.time()-t,flush=True)
