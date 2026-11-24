import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import *
from preflight import iq
import shutil
XS=[-10,-5,0,5,10,15,20]
def diskcheck():
    size=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
    assert size<=20*10**9,('disk limit',size)
def build(path,rs,fn,shape=None):
    marker=path.with_suffix('.done.json')
    if marker.exists():return
    path.parent.mkdir(exist_ok=True,parents=True);shape=shape or (len(rs),256,256)
    arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=shape)
    for count,i in enumerate(sorted(range(len(rs)),key=lambda i:(rs[i].get('block_file',''),int(rs[i].get('start',0))))):
        index=rs[i].get('_index',i);arr[index]=spec_fill(fn(rs[i]))
        if count%200==0:arr.flush();diskcheck();print(path.name,count,'/',len(rs),flush=True)
    arr.flush();del arr;diskcheck();dump(marker,dict(n=len(rs),shape=shape))
def main():
    start=time.time();assert (OUT/'preflight.json').exists()
    for folder in [OLD,BG]:folder.mkdir(exist_ok=True)
    for split in ['train','val','test','rejected_test']:
        name=f'2.4GHz_{split}_spec';rs=rows(HOLD/(name+'_meta.csv'));shutil.copyfile(HOLD/(name+'_meta.csv'),OLD/(name+'_meta.csv'))
        build(OLD/(name+'.npy'),rs,iq)
    br=rows(BOLD/'bgA_meta.csv');shutil.copyfile(BOLD/'bgA_meta.csv',BG/'bgA_meta.csv')
    selected=[dict(r,_index=i) for i,r in enumerate(br) if r['keep']=='True'];build(BG/'bgA_spec.npy',selected,iq,shape=(len(br),256,256))
    dump(BG/'keep_indices.json',[r['_index'] for r in selected])
    tr=rows(HOLD/'2.4GHz_train_spec_meta.csv');te=rows(HOLD/'2.4GHz_test_spec_meta.csv');lookup={r['sample_id']:r for r in tr+te+br}
    # Reuse exactly the persisted gains and pairing; background IQ is read-only.
    backgrounds={r['sample_id']:iq(r) for r in tr+te if r['label']=='background'}
    def mixed(r):
        b=backgrounds[r['background_sample_id']];return b+float(r['g'])*iq(lookup[r['sample_id']])
    for name in ['drone','bg']:
        rr=rows(OUT/f'mix_train_{name}.pairs.csv');rr=[dict(r,block_file=lookup[r['sample_id']]['block_file'],start=lookup[r['sample_id']]['start']) for r in rr]
        build(MIXCACHE/f'mix_train_{name}.npy',rr,mixed)
    for name in ['data_drone','v3_drone','v3_background']:
        src=[r for r in te if r['dataset']==('data' if name=='data_drone' else 'v3') and r['label']==('background' if name=='v3_background' else 'drone')];si={r['sample_id']:i for i,r in enumerate(src)}
        pairs=rows(HOLD/f'2.4GHz_mix_{name}.pairs.csv');shutil.copyfile(HOLD/f'2.4GHz_mix_{name}.pairs.csv',OLD/f'2.4GHz_mix_{name}.pairs.csv')
        rr=[dict(r,_index=(si[r['sample_id']],XS.index(int(r['x_db']))),block_file=lookup[r['sample_id']]['block_file'],start=lookup[r['sample_id']]['start']) for r in pairs]
        build(OLD/f'2.4GHz_mix_{name}.npy',rr,mixed,shape=(len(src),7,256,256))
    rr=[dict(r,_index=(i//7,i%7)) for i,r in enumerate(rows(OUT/'C1.pairs.csv'))];build(MIXCACHE/'C1.npy',rr,mixed,shape=(200,7,256,256))
    plan={r['block_file']:r for r in load(INPUT/'plan.json')};gs={r['group_id']:r['split'] for r in rows(INPUT/'group_split.csv')};qr=[]
    for run in load(ROOT/'work/bgA_20261109/quiet013_runs.json'):
        r=plan[run['block']]
        for j in range(run['start_ms'],run['end_ms']-19,20):qr.append(dict(sample_id=f'quiet013_{len(qr)}',block_file=run['block'],start=j*61440,n_samples=1228800,group_id=r['group_id'],split=gs[r['group_id']],training_label='drone' if gs[r['group_id']]=='train' else 'not_train',start_ms=j))
    assert len(qr)==19;writecsv(OUT/'quiet013_metadata.csv',qr);build(MIXCACHE/'quiet013.npy',qr,iq)
    rng=np.random.default_rng(20261109)
    build(MIXCACHE/'white_noise.npy',[{} for _ in range(80)],lambda r:(rng.standard_normal(1228800)+1j*rng.standard_normal(1228800)).astype(np.complex64))
    nn=np.load(ROOT/'work/review_20261109/notched_noise_2.4GHz.npy',mmap_mode='r');build(MIXCACHE/'notched_noise.npy',[dict(i=i) for i in range(80)],lambda r:np.array(nn[r['i']]))
    dump(OUT/'generation_done.json',dict(seconds=time.time()-start));print('GENERATION DONE',flush=True)
if __name__=='__main__':main()
