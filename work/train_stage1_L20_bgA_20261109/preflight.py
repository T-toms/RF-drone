import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from support import *
import hashlib
def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109'] for p in sorted((ROOT/'work'/d).rglob('*')) if p.is_file()}
if __name__=='__main__':
    dump(OUT/'protected_sha256_before.json',hashes())
    rr=rows(BG/'bgA_meta.csv');a=np.load(BG/'bgA_spec.npy',mmap_mode='r')
    counts=collections.Counter(r['split'] for r in rr if r['keep']=='True');assert dict(counts)==dict(train=2861,val=990,test=1287),counts
    gs={r['group_id']:r['split'] for r in rows(INPUT/'group_split.csv')}
    assert all(gs[r['group_id']]==r['split'] for r in rr)
    memberships=collections.defaultdict(set)
    for r in rr:memberships[r['group_id']].add(r['split'])
    for split in ['train','val','test']:
        for r in rows(OLD/f'2.4GHz_{split}_spec_meta.csv'):memberships[r['group_id']].add(split)
    assert all(len(v)==1 for v in memberships.values())
    checks=[]
    for split in ['train','val','test']:
        candidates=bgrows(split);rng=np.random.default_rng(20261109)
        for j in rng.choice(len(candidates),3,replace=False):
            i,r=candidates[j];z=np.load(r['block_file'],mmap_mode='r');lo=int(r['start']);v=spec(np.array(z[lo:lo+int(r['n_samples'])]),'2.4GHz');err=float(np.abs(v.astype(np.float32)-a[i].astype(np.float32)).max());assert err<=.01,err
            checks.append(dict(split=split,sample_id=r['sample_id'],max_abs_error=err))
    br=bgrows('test')[:200];m,ck=getmodel(OLD,2026);p=predict(m,torch.from_numpy(np.array(a[[i for i,r in br]])).unsqueeze(1))[:,1]
    ref={r['sample_id']:float(r['p_drone']) for r in rows(ROOT/'work/bgA_20261109/bgA_predictions.csv') if r['seed']=='2026'}
    err=max(abs(float(pd)-ref[r['sample_id']]) for pd,(i,r) in zip(p,br));assert err<1e-3,err
    result=dict(counts=dict(counts),split_mismatch=0,cross_split_overlap=0,pixel_checks=checks,prediction_max_error=err,protected_files=len(hashes()))
    dump(OUT/'preflight.json',result);print(json.dumps(result),flush=True)
