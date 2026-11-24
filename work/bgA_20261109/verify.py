from pathlib import Path
import json,csv,hashlib,numpy as np
O=Path(__file__).resolve().parent;P=O.parents[1]
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def verify():
 result={};source={r['sample_id']:r for r in rows(P/'work/crop_slice_20261107/index_L20.csv')};split={r['group_id']:r['split'] for r in rows(P/'work/crop_slice_20261107/group_split.csv')}
 for L in [10,20,30,40,50]:
  src=rows(P/f'work/crop_slice_20261107/index_L{L}.csv');by={r['sample_id']:r for r in src};r=rows(O/f'index_bgA_L{L}.csv');assert list(r[0])==list(src[0])+['paired_sample_id','mirror_shift_mhz'];assert len(r)==sum(v['dataset']=='data' and v['band']=='2.4GHz' for v in src);assert len({x['sample_id'] for x in r})==len(r)
  for v in r:
   s=by[v['paired_sample_id']];assert all(v[k]==s[k] for k in ['start','n_samples','group_id','split']);assert v['split']==split[v['group_id']];assert v['mirror_shift_mhz']=='0';assert v['dataset']=='dataA_mirror';assert all(v[k]=='False' for k in ['lack_feature','primary_weak','power_shift','floor_raised','interference'])
  result[str(L)]=dict(N=len(r),kept=sum(v['keep']=='True' for v in r),pairing=True,columns=True,split=True)
 before=json.loads((O/'hash_before.json').read_text(encoding='utf-8'));after={str(p.relative_to(P)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for d in ['crop_slice_20261107','train_stage1_L20_20261108'] for p in sorted((P/'work'/d).rglob('*')) if p.is_file()};changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k));(O/'hash_after.json').write_text(json.dumps(after,indent=2),encoding='utf-8');result['protected']=dict(before=len(before),after=len(after),changed=changed,identical=not changed);assert not changed,changed
 for name in ['bgA_predictions.csv','paired_drone_predictions.csv']:
  r=rows(O/name);assert {int(x['seed']) for x in r}=={2026,2027,2028};assert len(r)==result['20']['N']*3;result[name]=len(r)
 assert len(json.loads((O/'full_revised_final.json').read_text(encoding='utf-8')))==61;assert len(json.loads((O/'full_shape_profile.json').read_text(encoding='utf-8')))==61
 (O/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(result)
if __name__=='__main__':verify()
