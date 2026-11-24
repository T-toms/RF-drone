"""Independent artifact consistency checks; reads only the Q outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import psutil
from PIL import Image
O=Path(__file__).resolve().parent
proc=psutil.Process()
def check_memory():
 assert proc.memory_info().peak_wset < 8*1024**3
clips=json.loads((O/'clips.json').read_text(encoding='utf-8'))
ids={c['clip_id'] for c in clips}
assert len(ids)==len(clips)
assert all(abs(c['length']/c['fs']-.02)<1e-12 for c in clips)
assert all(Path(p['path']).drive.upper()=='G:' for c in clips for p in c['parts'])
assert all(sum(p['count'] for p in c['parts'])==c['length'] for c in clips)
assert all(c['fs'] in (122880000,250000000) for c in clips)
assert all('20260709-030521_ambient' not in c['group'] for c in clips)
assert all(n==6 for n in pd.DataFrame([c for c in clips if c['source']=='data']).groupby('group').size())
results=[]
for k in (6,10):
 for g in (.5,2):
  t=f'k{k}_g{g:g}'
  s=pd.read_csv(O/f'segments_{t}.csv');b=pd.read_csv(O/f'bands_{t}.csv');x=pd.read_csv(O/f'boxes_{t}.csv')
  assert len(s)==len(clips) and set(s.clip_id)==ids
  assert s.coverage.dropna().between(-1e-9,1+1e-9).all()
  assert s.loc[(s.label=='drone') & (s.reference_total>0),'coverage'].notna().all()
  assert s.loc[s.reference_total==0,'coverage'].isna().all()
  assert (x.area_px>=20).all() and (x.t_end_ms>x.t_start_ms).all()
  assert (x.t_start_ms>=0).all() and (x.t_end_ms<=20).all()
  assert (x.f_end_mhz>x.f_start_mhz).all()
  assert np.allclose(b.width_mhz,b.f_end_mhz-b.f_start_mhz)
  assert b.time_duty.between(0,1).all()
  assert not b.duplicated(['clip_id','band_id']).any()
  bx={i:v for i,v in x.groupby('clip_id')}
  nb=b.groupby('clip_id').size().to_dict(); nx=x.groupby('clip_id').size().to_dict()
  for _,v in s.iterrows():
   assert v.band_count==nb.get(v.clip_id,0) and v.box_count==nx.get(v.clip_id,0)
  for cid,bs in b.groupby('clip_id'):
   bs=bs.sort_values('band_id'); assert np.array_equal(bs.band_id.to_numpy(),np.arange(1,len(bs)+1))
   assert np.all(bs.f_start_mhz.to_numpy()[1:]-bs.f_end_mhz.to_numpy()[:-1]>=g-1e-8)
   xx=bx[cid]; assigned=np.zeros(len(xx),int)
   for _,v in bs.iterrows():
    inside=(xx.f_start_mhz>=v.f_start_mhz-1e-8)&(xx.f_end_mhz<=v.f_end_mhz+1e-8)
    assert inside.sum()==v.box_count
    assigned+=inside.to_numpy()
   assert (assigned==1).all()
  results.append(dict(parameter=t,clips=len(s),boxes=len(x),bands=len(b),passed=True))
  check_memory()
pi=pd.read_csv(O/'plot_index.csv')
assert pi.groupby(['parameter','category','page']).size().max()<=8
for p,v in pi.groupby('parameter'):
 assert (v.category=='lowest_coverage').sum()==4
 bgids=v.loc[v.category=='raw_background','clip_id'].tolist()
 byid={c['clip_id']:c for c in clips}
 assert len(bgids)==6
 for scene in ('07-09城区','07-10实验室','07-11家里'):
  sceneclips=[byid[cid] for cid in bgids if byid[cid]['scene']==scene]
  assert len(sceneclips)==2 and len({c['parts'][0]['path'] for c in sceneclips})==2
 for src in ('data','raw'):
  expected=len(set((c['model'],c['band']) for c in clips if c['source']==src and c['label']=='drone'))
  assert (v.category==src+'_drone').sum()==expected
assert len((O/'REPORT.md').read_text(encoding='utf-8').splitlines())<=80
assert json.loads((O/'protected_before.json').read_text(encoding='utf-8'))==json.loads((O/'protected_after.json').read_text(encoding='utf-8'))
pngs=list(O.glob('图_*.png'))
for p in pngs:
 with Image.open(p) as im: im.verify()
check_memory()
result=dict(passed=True,parameters=results,plot_rows=len(pi),pngs_verified=len(pngs),peak_bytes=proc.memory_info().peak_wset,report_lines=len((O/'REPORT.md').read_text(encoding='utf-8').splitlines()))
(O/'qa_artifacts.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
