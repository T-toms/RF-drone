from pathlib import Path
import json,hashlib,collections
import h5py,numpy as np
R=Path('G:/drone');O=Path(__file__).parent/'audit_drone_20261023'
src=json.loads((O/'raw_json_complete.json').read_text(encoding='utf8')); byhash=collections.defaultdict(list)
for x in src:byhash[x['metadata']['capture']['sha256']].append(x)
chosen=[x for xs in byhash.values() if len(xs)>1 for x in xs]
chosen.extend(next(x for x in src if x['metadata']['capture']['sample_rate_hz']==125000000) for _ in range(1))
out=[]
for x in chosen:
 p=(R/x['path']).with_suffix('.bin'); h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 out.append(dict(path=str(p),actual=h.hexdigest(),matches_json=h.hexdigest()==x['metadata']['capture']['sha256']))
(O/'raw_selected_sha256.json').write_text(json.dumps(out,indent=2),encoding='utf8')
comp=[]
for p in R.rglob('*.mat'):
 if 'raw' not in p.parts or p.name=='bg_feature_cache.mat':continue
 b=p.with_suffix('.bin')
 with h5py.File(p,'r') as f:
  iq=f['IQ'];n=iq.shape[0];matches=[];ratios=[]
  for start in [0,n//2,n-4096]:
   a=iq[start:start+4096,0];z=a['real']+1j*a['imag']
   v=np.fromfile(b,dtype='<i2',count=8192,offset=start*4).reshape(-1,2);c=v[:,0].astype(float)+1j*v[:,1].astype(float)
   matches.append(bool(np.array_equal(z,c)));ratios.append(float(np.vdot(c,z).real/np.vdot(c,c).real))
  comp.append(dict(path=str(p),fs=float(f['fs'][0,0]),fc=float(f['centfreq'][0,0]),n=n,head_middle_tail_equal=matches,scale=ratios))
(O/'raw_mat_bin_comparison.json').write_text(json.dumps(comp,indent=2),encoding='utf8')
print('DONE',len(out),'hashes',sum(x['matches_json'] for x in out), 'MAT comparisons',len(comp),flush=True)
