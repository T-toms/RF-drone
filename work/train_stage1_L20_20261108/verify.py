import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import hashlib
def run():
    checks={}; hashes=load(OUT/'input_sha256.json')
    checks['original_inputs_unchanged']=all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items());assert checks['original_inputs_unchanged']
    index=rows(INPUT/'index_L20.csv');checks['cache_counts']={};checks['val_threshold_checks']=[]
    for b in BANDS:
        for split in ['train','val','test']:
            rs=meta(b,split);expected={r['sample_id'] for r in index if r['band']==b and r['split']==split and r['keep']=='True'};assert expected=={r['sample_id'] for r in rs};a=np.load(cachepath(b,split),mmap_mode='r');assert a.shape==(len(rs),256,256);assert a.dtype==np.float16
            assert np.isfinite(a).all();checks['cache_counts'][b+'_'+split]=len(rs)
        vx,vy,vr=dataset(b,'val');yy=vy.numpy()
        for s in SEEDS:
            logs=rows(OUT/f'train_log_{b}_seed{s}.csv');assert len(logs)==40
            chosen=min(logs,key=lambda r:(float(r['val_fa'])+float(r['val_miss']),float(r['val_loss'])));m,c=model(b,s);assert c['epoch']==int(chosen['epoch']);p=predict(m,vx)[:,1];fa=float((p[yy==0]>=c['thr_fa1']).mean());assert fa<=.01,(b,s,fa)
            checks['val_threshold_checks'].append(dict(band=b,seed=s,n_background=int((yy==0).sum()),threshold=c['thr_fa1'],fa=fa))
            predictions=rows(OUT/f'test_predictions_{b}_seed{s}.csv');assert len(predictions)==checks['cache_counts'][b+'_test'];assert all(r['keep']=='True' for r in predictions)
            del m;torch.cuda.empty_cache()
    pairs=rows(OUT/'mix_pairs.csv');assert max(abs(float(r['achieved_ratio_db'])-float(r['x_db'])) for r in pairs)<1e-8
    for b in BANDS:
        rs=meta(b,'test')
        for group in ['data_drone','v3_drone','v3_background']:
            ds,lab=group.split('_');expect={r['sample_id'] for r in rs if r['dataset']==ds and r['label']==lab};actual=[r for r in pairs if r['band']==b and r['group']==group];assert len(actual)==len(expect)*7;assert {r['sample_id'] for r in actual}==expect
        for prefix in ['fig0_input_examples','fig2_recall_by_source','fig3_mix_test','fig3b_mix_examples','fig4_gradcam','fig4b_gradcam_mean','fig5_curves']:assert (OUT/f'{prefix}_{b}.png').is_file()
    checks['training_runs']=6;checks['epochs_per_run']=40;checks['mix_pairs']=len(pairs);checks['mix_power_ratio_max_error_db']=max(abs(float(r['achieved_ratio_db'])-float(r['x_db'])) for r in pairs);checks['passed']=True;dump(OUT/'verification.json',checks);print('VERIFIED',checks,flush=True)
if __name__=='__main__':run()
