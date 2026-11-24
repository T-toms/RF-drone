import sys
sys.dont_write_bytecode=True
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O))
from support import *

def run(final=True):
    rng=np.random.default_rng(20261110);tr=rows(OLD/'2.4GHz_train_spec_meta.csv');backgrounds=[r for r in tr if r['dataset']=='v3' and r['label']=='background']
    result={}
    for name,src in [('drone',[r for r in tr if r['dataset']=='data' and r['label']=='drone']),('bg',[r for i,r in bgrows('train')])]:
        rs=rows(OUT/f'mix_train_{name}.pairs.csv');a=np.load(MIXCACHE/f'mix_train_{name}.npy',mmap_mode='r');assert a.shape==(len(src),256,256) and a.dtype==np.float16
        for i,r in enumerate(rs):
            x=rng.uniform(-5,15);j=int(rng.integers(234));assert r['sample_id']==src[i]['sample_id'] and r['background_sample_id']==backgrounds[j]['sample_id'] and abs(float(r['X'])-x)<1e-12
            assert abs(float(r['actual_power_ratio_db'])-x)<1e-10
        for chunk in range(0,len(a),64):assert np.isfinite(a[chunk:chunk+64]).all()
        result[name]=dict(shape=list(a.shape),rng_match=True,finite=True)
    if final:
        assert load(OUT/'integrity.json')['identical']
        for v in ['La','Lb']:
            for s in SEEDS:
                ck=torch.load(DEST/f'2.4GHz_{v}_seed{s}_best.pt',map_location='cpu',weights_only=False)
                n=np.array(ck['train_counts']);w=np.array(ck['sample_weights']);mass=n*w/n.sum();expected=[.25,.25,.25,.25] if v=='La' else [.25,.25,.125,.25,.125]
                assert np.max(np.abs(mass-expected))<1e-6 and load(DEST/f'2.4GHz_{v}_seed{s}_done.json')['epochs']==40
                assert ck['thr_fa1']==max(ck['thr_v3'],ck['thr_bgA'])
                assert len(rows(OUT/f'train_log_{v}_seed{s}.csv'))==40
                rs=rows(OUT/f'test_predictions_2.4GHz_{v}_seed{s}.csv');assert len(rs)==len(rows(OLD/'2.4GHz_test_spec_meta.csv'))
                assert len(rows(OUT/f'bgA_test_predictions_{v}_seed{s}.csv'))==1287
                assert len(rows(OUT/f'C1_predictions_{v}_seed{s}.csv'))==1400
        cr=rows(OUT/'C1.pairs.csv');assert len(cr)==1400 and len({r['sample_id'] for r in cr})==200
        assert all(r['split']=='test' for r in cr)
        for name in ['metrics_paired.csv','mix_data_model_paired.csv','wifi_group_descriptors.csv','fig3_mix_test_2.4GHz.png','fig_C1_2.4GHz.png','fig5_curves_2.4GHz.png','图14_叠加WiFi训练前后_2.4GHz.png']:assert (OUT/name).stat().st_size>0
        assert len((OUT/'REPORT.md').read_text(encoding='utf-8').splitlines())<=100
    dump(OUT/('acceptance.json' if final else 'generation_acceptance.json'),dict(passed=True,final=final,checks=result))
    print('ACCEPTANCE PASS',result,flush=True)
if __name__=='__main__':run('--generation-only' not in sys.argv)
