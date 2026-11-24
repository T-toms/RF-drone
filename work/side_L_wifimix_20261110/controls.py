import sys
sys.dont_write_bytecode=True
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O))
from support import *
from preflight import iq,mixing

def run():
    if (MIXCACHE/'C1_done.json').exists():return
    test=rows(OLD/'2.4GHz_test_spec_meta.csv');bg=[r for r in test if r['label']=='background'];assert len(bg)==80
    br=bgrows('test');rng=np.random.default_rng(20261111);chosen=[br[i][1] for i in rng.choice(len(br),200,replace=False)]
    bs=[iq(r) for r in bg];xs=[-10,-5,0,5,10,15,20];pairs=[]
    arr=np.lib.format.open_memmap(MIXCACHE/'C1.npy',mode='w+',dtype=np.float16,shape=(200,7,256,256))
    for i,r in enumerate(chosen):
        d=iq(r)
        for k,x in enumerate(xs):
            z,g,pd,pb=mixing(d,bs[i%80],x);arr[i,k]=spec(z,'2.4GHz');pairs.append(dict(r,x_db=x,background_sample_id=bg[i%80]['sample_id'],background_group_id=bg[i%80]['group_id'],g=g,P_d=pd,P_b=pb))
        if i%40==0:print('C1',i,flush=True)
    arr.flush();writecsv(OUT/'C1.pairs.csv',pairs);dump(MIXCACHE/'C1_done.json',dict(n=200,x_db=xs))
if __name__=='__main__':run()
