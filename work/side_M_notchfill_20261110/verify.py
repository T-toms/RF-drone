import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import *
import hashlib
def run():
    checks=[]
    for name in ['mix_train_drone.pairs.csv','mix_train_bg.pairs.csv','C1.pairs.csv','train.py']:
        assert (OUT/name).read_bytes()==(L/name).read_bytes(),name
    for path in sorted(DEST.rglob('*.npy')):
        a=np.load(path,mmap_mode='r');assert a.dtype==np.float16 and a.shape[-2:]==(256,256),(path,a.shape,a.dtype)
        for start in range(0,len(a),100):assert np.isfinite(a[start:start+100]).all(),path
        checks.append(dict(path=str(path),shape=list(a.shape),dtype=str(a.dtype),finite=True))
    for split in ['train','val','test','rejected_test']:
        stem=f'2.4GHz_{split}_spec';assert (HOLD/(stem+'_meta.csv')).read_bytes()==(OLD/(stem+'_meta.csv')).read_bytes()
        assert len(np.load(OLD/(stem+'.npy'),mmap_mode='r'))==len(rows(OLD/(stem+'_meta.csv')))
    assert (BOLD/'bgA_meta.csv').read_bytes()==(BG/'bgA_meta.csv').read_bytes()
    assert len(np.load(MIXCACHE/'mix_train_drone.npy',mmap_mode='r'))==2832
    assert len(np.load(MIXCACHE/'mix_train_bg.npy',mmap_mode='r'))==2861
    size=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file());assert size<20*10**9
    dump(OUT/'cache_acceptance.json',dict(arrays=checks,bytes=size,pairs_identical=True,train_identical=True))
    if '--final' in sys.argv:
        for s in SEEDS:
            assert len(rows(OUT/f'train_log_Lb_seed{s}.csv'))==40
            for name in ['test_predictions_2.4GHz','bgA_test_predictions','bgA_val_predictions','rejected_predictions','quiet013_predictions','white_noise_predictions','notched_noise_predictions']:
                for tag in ['Lb','M']:
                    rs=rows(OUT/f'{tag}_{name}_seed{s}.csv');assert rs
                    assert all(0<=float(r['p_drone'])<=1 for r in rs)
        assert len((OUT/'REPORT.md').read_text(encoding='utf-8').splitlines())<=90
        assert load(OUT/'integrity.json')['baseline_identical']
        dump(OUT/'acceptance.json',dict(caches=True,training_runs=3,epochs=120,smoke_epochs=3,protected_baseline_identical=True,report=True))
    print('VERIFIED',len(checks),'arrays',size,flush=True)
if __name__=='__main__':run()
