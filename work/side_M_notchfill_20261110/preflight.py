import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import *
import fill, hashlib, subprocess
PROTECTED=['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109','train_stage1_L20_bgA_20261109','side_L_wifimix_20261110','review_20261109']
def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in PROTECTED for p in sorted((ROOT/'work'/d).rglob('*')) if p.is_file()}
def iq(r):
    a=np.load(Path(r['block_file']),mmap_mode='r');lo=int(r['start']);return np.array(a[lo:lo+int(r['n_samples'])])
def main():
    gpu=subprocess.check_output(['nvidia-smi'],text=True,encoding='utf-8');(OUT/'gpu_before.txt').write_text(gpu,encoding='utf-8');assert 'python' not in gpu.lower(),gpu
    if not (OUT/'protected_before.json').exists():dump(OUT/'protected_before.json',hashes())
    tr=rows(HOLD/'2.4GHz_train_spec_meta.csv');te=rows(HOLD/'2.4GHz_test_spec_meta.csv');br=rows(BOLD/'bgA_meta.csv')
    counts=[sum(r['dataset']==d and r['label']==lab for r in tr) for d,lab in [('data','drone'),('v3','drone'),('v3','background')]]+[sum(r['keep']=='True' and r['split']==s for r in br) for s in ['train','val','test']]+[len(rows(OUT/f'mix_train_{n}.pairs.csv')) for n in ['drone','bg']]
    result={'counts':counts};assert counts==[2832,152,234,2861,990,1287,2832,2861],counts
    rng=np.random.default_rng(20261110);btest=[r for r in br if r['keep']=='True' and r['split']=='test']
    mask=np.ones(256,dtype=bool)
    for idx in fill.hole_idx:
        mask &= ~np.array([any(a<=j<b for j in idx) for a,b in zip(*fill.edges)])
    result['consistency']=[]
    for name,rs in [('H',te),('bgA',btest)]:
        for i in rng.choice(len(rs),3,replace=False):
            z=iq(rs[i]);old=spec_original(z,'2.4GHz');saved=fill.hole_idx;fill.hole_idx=[]
            unfilled=spec_fill(z);fill.hole_idx=saved
            assert np.array_equal(old,unfilled),'unfilled mismatch'
            new=spec_fill(z);x=old[mask].astype(float).ravel();y=new[mask].astype(float).ravel();a,b=np.polyfit(x,y,1);res=float(np.max(np.abs(y-a*x-b)))
            new32=spec_fill(z,dtype=np.float32);fill.hole_idx=[];old32=spec_fill(z,dtype=np.float32);fill.hole_idx=saved
            xx=old32[mask].ravel();yy=new32[mask].ravel();aa,bb=np.polyfit(xx,yy,1);res32=float(np.max(np.abs(yy-aa*xx-bb)))
            result['consistency'].append(dict(source=name,sample_id=rs[i]['sample_id'],identical=True,residual_float16=res,residual_float32=res32));assert res<=.02 and res32<=.001,result
    lookup={r['sample_id']:r for r in tr+br};result['mix_recompute']=[]
    for name in ['drone','bg']:
        arr=np.load(LDEST/'cache'/f'mix_train_{name}.npy',mmap_mode='r')
        for i,r in enumerate(rows(OUT/f'mix_train_{name}.pairs.csv')[:3]):
            z=iq(lookup[r['background_sample_id']])+float(r['g'])*iq(lookup[r['sample_id']]);diff=float(np.max(np.abs(spec_original(z,'2.4GHz').astype(float)-arr[i].astype(float))))
            result['mix_recompute'].append(diff);assert diff<=.01,result
    result['row_stats']=[];rng=np.random.default_rng(20261110)
    nn=np.load(ROOT/'work/review_20261109/notched_noise_2.4GHz.npy',mmap_mode='r')
    for name,zs in [('bgA',[iq(btest[i]) for i in sorted(rng.choice(len(btest),40,replace=False))]),('notched',[np.array(nn[i]) for i in range(40)])]:
        imgs=[spec_fill(z) for z in zs]
        for n in fill.HOLES:
            d,s=fill.row_stats(imgs,n);result['row_stats'].append(dict(source=name,freq=n,mean_difference=d,std_ratio=s));assert abs(d)<=.2 and .85<=s<=1.15,result
    dump(OUT/'preflight.json',result);print(result,flush=True)
if __name__=='__main__':main()
