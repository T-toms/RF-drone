import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from support import *
from evaluate import cp
from preflight import hashes
def run():
    checks=[]
    original=rows(OLD/'2.4GHz_test_spec_meta.csv');bgr=[r for i,r in bgrows('test')]
    for s in SEEDS:
        logs=rows(OUT/f'train_log_2.4GHz_seed{s}.csv');assert [int(r['epoch']) for r in logs]==list(range(1,41))
        best=min(logs,key=lambda r:(float(r['val_fa_v3'])+float(r['val_fa_bgA'])+float(r['val_miss']),float(r['val_loss'])))
        ck=torch.load(DEST/f'2.4GHz_seed{s}_best.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==int(best['epoch']);assert ck['thr_fa1']==max(ck['thr_v3'],ck['thr_bgA'])
        assert ck['val_v3_background_n']==80 and ck['val_bgA_background_n']==990
        for name,ref in [(f'test_predictions_2.4GHz_seed{s}.csv',original),(f'bgA_test_predictions_seed{s}.csv',bgr)]:
            rs=rows(OUT/name);assert [r['sample_id'] for r in rs]==[r['sample_id'] for r in ref]
            assert all(np.isfinite(float(r['p_drone'])) and 0<=float(r['p_drone'])<=1 and int(r['pred_thr'])==int(float(r['p_drone'])>=ck['thr_fa1']) for r in rs)
        checks.append(dict(seed=s,epochs=len(logs),best_epoch=ck['epoch'],test_n=len(original),bgA_test_n=len(bgr),threshold=ck['thr_fa1']))
    metrics=rows(OUT/'metrics_all.csv');assert all(len([r for r in metrics if r['round']==tag and int(r['seed'])==s and r['mode']==mode and r['category']=='main'])==10 for tag in ['H','J'] for s in SEEDS for mode in ['argmax','thr_fa1'])
    assert len(rows(OUT/'mix_results.csv'))==2*3*2*3*7
    assert abs(cp(0,80)-(1-.05**(1/80)))<1e-12 and cp(80,80)==1
    for r in metrics:
        if r['fa']=='True':assert float(r['rate'])<=float(r['cp95_upper'])<=1
    assert hashes()==load(OUT/'protected_sha256_before.json')
    assert len((OUT/'REPORT.md').read_text(encoding='utf-8').splitlines())<=90
    for name in ['图12_加入接收机A背景前后_2.4GHz.png','fig3_mix_test_2.4GHz.png','fig5_curves_2.4GHz.png']:assert (OUT/name).stat().st_size>10000
    dump(OUT/'acceptance.json',dict(passed=True,seeds=checks,metric_rows=len(metrics),mix_rows=252,protected_sha256_identical=True))
    print('ACCEPTANCE PASSED',checks,flush=True)
if __name__=='__main__':run()
