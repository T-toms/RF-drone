from pathlib import Path
import json,hashlib,time,csv,collections
import numpy as np,h5py
from scipy.io import loadmat
O=Path(__file__).parent/'audit_drone_20261023';R=Path('G:/drone')
src=json.loads((O/'raw_json_complete.json').read_text(encoding='utf8'))
out=O/'raw_full_hash.jsonl';done={}
if out.exists():done={r['path']:r for r in map(json.loads,out.read_text(encoding='utf8').splitlines())}
with out.open('a',encoding='utf8') as f:
 for i,x in enumerate(src):
  rel=x['path'];p=(R/rel).with_suffix('.bin');key=str(p)
  if key in done:continue
  row={'path':key,'expected':x['metadata']['capture']['sha256']};h=hashlib.sha256();n=0;zeros=0;saturated=0
  try:
   with p.open('rb') as b:
    for block in iter(lambda:b.read(8*1024*1024),b''):
     h.update(block);n+=len(block);packed=np.frombuffer(block,dtype='<i4');zeros+=int(packed.size-np.count_nonzero(packed));v=np.frombuffer(block,dtype='<i2');saturated+=int(np.count_nonzero((v==32767)|(v==-32768)))
   row.update(bytes=n,actual=h.hexdigest(),matches=h.hexdigest()==row['expected'],zero_iq_samples=zeros,rail_components=saturated)
  except Exception as e:row['error']=str(e)
  f.write(json.dumps(row)+'\n');f.flush()
  if i%50==0:print('BIN',i,'/890',flush=True)
rows=list(map(json.loads,out.read_text(encoding='utf8').splitlines()));(O/'raw_full_hash_summary.json').write_text(json.dumps({'files':len(rows),'matches':sum(r.get('matches',False) for r in rows),'errors':[r for r in rows if 'error'in r],'mismatches':[r for r in rows if r.get('matches') is False]},indent=2),encoding='utf8')
comparisons=[]
for p in R.joinpath('raw').rglob('*.mat'):
 if not h5py.is_hdf5(p):continue
 equal=True;n=0
 with h5py.File(p,'r') as f,p.with_suffix('.bin').open('rb') as b:
  iq=f['IQ'];total=iq.shape[0]
  for start in range(0,total,1000000):
   z=iq[start:min(start+1000000,total),0];v=np.frombuffer(b.read(len(z)*4),dtype='<i2').reshape(-1,2)
   equal=equal and bool(np.array_equal(z['real'],v[:,0]/32768.) and np.array_equal(z['imag'],v[:,1]/32768.));n+=len(z)
 comparisons.append({'path':str(p),'samples':n,'all_samples_equal_after_div32768':equal});print('MAT comparison',len(comparisons),flush=True)
(O/'raw_mat_full_comparison.json').write_text(json.dumps(comparisons,indent=2),encoding='utf8')
(O/'continued_complete.json').write_text(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'bins':len(rows),'mat':len(comparisons)},indent=2),encoding='utf8')
print('COMPLETE',flush=True)
