import sys
sys.dont_write_bytecode=True
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O))
from support import *
from preflight import iq,mixing,power

def generate():
    start=time.time()
    if (MIXCACHE/'generation_done.json').exists():return
    tr=rows(OLD/'2.4GHz_train_spec_meta.csv');br=rows(BG/'bgA_meta.csv')
    backgrounds=[r for r in tr if r['dataset']=='v3' and r['label']=='background']
    bs=[iq(r) for r in backgrounds];assert len(bs)==234
    rng=np.random.default_rng(20261110)
    summary={}
    for name,sources in [('drone',[r for r in tr if r['dataset']=='data' and r['label']=='drone']),('bg',[r for i,r in bgrows('train')])]:
        assignments=[(float(rng.uniform(-5,15)),int(rng.integers(0,234))) for r in sources]
        arr=np.lib.format.open_memmap(MIXCACHE/f'mix_train_{name}.npy',mode='w+',dtype=np.float16,shape=(len(sources),256,256))
        pairs=[None]*len(sources)
        for count,i in enumerate(sorted(range(len(sources)),key=lambda i:(sources[i]['block_file'],int(sources[i]['start'])))):
            r=sources[i];x,j=assignments[i];d=iq(r);z,g,pd,pb=mixing(d,bs[j],x);arr[i]=spec(z,'2.4GHz')
            pairs[i]=dict(sample_id=r['sample_id'],background_sample_id=backgrounds[j]['sample_id'],background_group_id=backgrounds[j]['group_id'],X=x,g=g,P_d=pd,P_b=pb,actual_power_ratio_db=10*np.log10(g*g*pd/pb))
            if count%200==0:print('GENERATE',name,count,'seconds',time.time()-start,flush=True);arr.flush()
        arr.flush();writecsv(OUT/f'mix_train_{name}.pairs.csv',pairs)
        summary[name]=dict(n=len(sources),X_quantiles=np.quantile([r['X'] for r in pairs],[0,.25,.5,.75,1]).tolist(),max_power_ratio_error=max(abs(r['X']-r['actual_power_ratio_db']) for r in pairs))
        del arr
    dump(MIXCACHE/'generation_done.json',dict(seconds=time.time()-start,summary=summary))
if __name__=='__main__':generate()
