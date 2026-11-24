"""Read-only source audit; artifacts are written beside this script."""
from pathlib import Path
import os,json,csv,collections,re,hashlib,time
import numpy as np
from scipy.io import whosmat,loadmat
ROOT=Path('G:/drone'); OUT=Path(__file__).parent/'audit_drone_20261023'; OUT.mkdir(exist_ok=True)
errors=[]
def dump(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def csvout(n,rows):
 if not rows:return
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
files=[]
for d,ds,fs in os.walk(ROOT,onerror=lambda e:errors.append(str(e))):
 for name in fs:
  p=Path(d)/name
  try: files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,ext=p.suffix.lower()))
  except Exception as e:errors.append(str(e))
csvout('all_files.csv',files)
groups=collections.defaultdict(lambda:[0,0])
for r in files:
 k=(r['path'].split('\\')[0],r['ext']);groups[str(k)][0]+=1;groups[str(k)][1]+=r['bytes']
dump('totals.json',dict(groups)); print('TOTALS',dict(groups),flush=True)
flatvals=collections.defaultdict(collections.Counter); raw=[]; source=[]
def flatten(x,p=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from flatten(v,p+'.'+k if p else k)
 else:yield p,x
for r in files:
 if r['ext']!='.json':continue
 p=ROOT/r['path']
 try:
  m=json.loads(p.read_text(encoding='utf-8-sig'));source.append(dict(path=r['path'],metadata=m))
  for k,v in flatten(m):flatvals[k][json.dumps(v,ensure_ascii=False,sort_keys=True)]+=1
  c=m.get('capture',{});s=m.get('session',{});dec=m.get('declared',{}); b=p.with_suffix('.bin')
  row=dict(path=r['path'],bin_exists=b.exists(),bytes=b.stat().st_size if b.exists() else None,**{k:c.get(k) for k in ['datatype','sample_rate_hz','center_freq_hz','usable_bw_hz','sample_count','t_start_utc','sha256']},session=s.get('id'),seq=s.get('seq'),campaign=s.get('campaign'),paired_bg=s.get('paired_bg'),**{k:json.dumps(dec.get(k),ensure_ascii=False) for k in ['class','model','state','links','bw','fc','rf_env']},notes=m.get('notes'))
  row['size_ok']=row['bytes']==c.get('sample_count',-1)*4 if c.get('datatype')=='ci16_le' else None
  row['duration_s']=c['sample_count']/c['sample_rate_hz'];raw.append(row)
 except Exception as e:errors.append(dict(path=str(p),error=str(e)))
csvout('raw_metadata.csv',raw);dump('raw_json_complete.json',source);dump('raw_field_values.json',{k:dict(v) for k,v in flatvals.items()})
ids={r['session'] for r in raw}; checks=[]
for r in raw:
 try:
  b=(ROOT/r['path']).with_suffix('.bin')
  with b.open('rb') as f: head=f.read(4096);f.seek(-min(4096,r['bytes']),2);tail=f.read()
  checks.append(dict(path=str(b),head_bytes=len(head),tail_bytes=len(tail),size_ok=r['size_ok'],paired_bg_exists=r['paired_bg'] in ids if r['paired_bg'] else None))
 except Exception as e:errors.append(dict(path=r['path'],error=str(e)))
dump('raw_read_checks.json',checks)
sessions=[]
for sid in sorted(ids):
 rr=[r for r in raw if r['session']==sid]; rr.sort(key=lambda r:r['seq']); gaps=[]
 from datetime import datetime
 for a,b in zip(rr,rr[1:]):
  gap=(datetime.fromisoformat(b['t_start_utc'].replace('Z','+00:00'))-datetime.fromisoformat(a['t_start_utc'].replace('Z','+00:00'))).total_seconds()-a['duration_s'];gaps.append(gap)
 sessions.append(dict(session=sid,files=len(rr),campaign=rr[0]['campaign'],cls=rr[0]['class'],model=rr[0]['model'],paired_bg=rr[0]['paired_bg'],paired_exists=rr[0]['paired_bg'] in ids if rr[0]['paired_bg'] else None,frequencies=sorted({r['center_freq_hz'] for r in rr}),rates=sorted({r['sample_rate_hz'] for r in rr}),notes=sorted({r['notes'] for r in rr}),gap_seconds=gaps,paths=[r['path'] for r in rr]))
dump('raw_sessions.json',sessions)
print('RAW',len(raw),'sessions',len(sessions),'size failures',sum(not r['size_ok'] for r in raw),flush=True)
mats=[]; parsed=[]; dup=[]; canon={}
for r in files:
 if r['ext']!='.mat':continue
 p=ROOT/r['path']; stem=re.sub(r'(\(1\)|_\d{8}_\d{6})$','',p.stem);canon.setdefault((str(p.parent),stem),[]).append(p)
 if r['path'].startswith('data'):
  m=re.search(r'-(\d{4}-\d\d-\d\d-\d\d-\d\d-\d\d)-epoch-(\d+)_seg(\d+)',p.name)
  if m:parsed.append(dict(path=r['path'],folder=p.parent.name,split=p.parts[3],session=m[1],epoch=int(m[2]),seg=int(m[3]),duplicate=stem!=p.stem))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
for k,ps in canon.items():
 if len(ps)>1:
  hh=[(str(p.relative_to(ROOT)),sha(p)) for p in ps];dup.append(dict(files=hh,equal=len({h for p,h in hh})==1))
dump('duplicate_hashes.json',dup);csvout('data_index.csv',parsed);print('DUP',len(dup),flush=True)
# Inspect variable headers in every MAT; full numerical load once per recording folder and all raw MATs.
seen=set(); numerical=[]
for i,r in enumerate(x for x in files if x['ext']=='.mat'):
 p=ROOT/r['path']
 try:
  vars=whosmat(p);mats.append(dict(path=r['path'],variables=vars))
  key=str(p.parent)
  if key not in seen or r['path'].startswith('raw'):
   seen.add(key); d=loadmat(p); row=dict(path=r['path'],fields={})
   for k,v in d.items():
    if k.startswith('__'):continue
    z=dict(shape=v.shape,dtype=str(v.dtype))
    if np.issubdtype(v.dtype,np.number):
     z.update(finite=bool(np.isfinite(v).all()),min_real=float(v.real.min()) if v.size else None,max_real=float(v.real.max()) if v.size else None)
     if v.size<=32:z['values']=str(v)
    else:z['preview']=str(v)[:2000]
    row['fields'][k]=z
   numerical.append(row)
 except Exception as e:errors.append(dict(path=r['path'],error=str(e)))
 if i%500==0:print('MAT',i,flush=True)
dump('mat_headers.json',mats);dump('mat_numerical_samples.json',numerical);dump('errors.json',errors)
dump('completion.json',dict(files=len(files),json=len(raw),mats=len(mats),numerical_files=len(numerical),duplicate_groups=len(dup),errors=len(errors),finished=time.strftime('%Y-%m-%d %H:%M:%S')))
print('DONE',len(errors),flush=True)
