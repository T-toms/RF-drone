from pathlib import Path
import csv,json,time
import numpy as np
from scipy.io import loadmat
from concurrent.futures import ThreadPoolExecutor
O=Path(__file__).parent/'audit_drone_20261023';R=Path('G:/drone')
rs=list(csv.DictReader((O/'data_index.csv').open(encoding='utf-8-sig')));rs=[r for r in rs if r['duplicate']=='False'];bad=[]
def inspect(r):
  out={'path':r['path']}
  try:
   v=loadmat(R/r['path'])['data'].ravel();nz=np.flatnonzero(v);out.update(n=int(v.size),dtype=str(v.dtype),finite=bool(np.isfinite(v).all()),nonzero=int(nz.size),first_nonzero=int(nz[0]) if nz.size else None,last_nonzero=int(nz[-1]) if nz.size else None,max_abs=float(np.abs(v).max()),integer_iq=bool(np.all(v.real==np.rint(v.real)) and np.all(v.imag==np.rint(v.imag))))
   out['zero_blocks_61440']=[k for k in range(100) if not np.any(v[k*61440:(k+1)*61440])]
  except Exception as e:out['error']=str(e)
  return out
existing=[json.loads(s) for s in (O/'data_values.jsonl').read_text(encoding='utf8').splitlines()] if (O/'data_values.jsonl').exists() else []
done={r['path'] for r in existing};pending=[r for r in rs if r['path'] not in done]
with (O/'data_values.jsonl').open('a',encoding='utf8') as f, ThreadPoolExecutor(max_workers=2) as pool:
 for i,out in enumerate(pool.map(inspect,pending),len(done)+1):
  if 'error' in out:bad.append(out)
  f.write(json.dumps(out,ensure_ascii=False)+'\n');f.flush()
  if i%200==0:print('VALUES',i,'/',len(rs),flush=True)
(O/'values_complete.json').write_text(json.dumps({'files':len(rs),'errors':bad,'completed':time.strftime('%Y-%m-%d %H:%M:%S')},indent=2),encoding='utf8')
print('VALUES DONE',len(rs),flush=True)
