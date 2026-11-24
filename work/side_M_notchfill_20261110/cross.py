import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import *
from evaluate import tensor,cp
import fill
def run():
    start=time.time();out=[]
    ti=rows(OLD/'2.4GHz_test_spec_meta.csv');br=bgrows('test');ii=[i for i,r in br]
    oldbg=tensor(np.load(BOLD/'bgA_spec.npy',mmap_mode='r')[ii]);newbg=tensor(np.load(BG/'bgA_spec.npy',mmap_mode='r')[ii])
    oldnn=tensor([spec_original(np.array(z),'2.4GHz') for z in np.load(ROOT/'work/review_20261109/notched_noise_2.4GHz.npy',mmap_mode='r')])
    newnn=tensor(np.load(MIXCACHE/'notched_noise.npy'));newtest=tensor(np.load(OLD/'2.4GHz_test_spec.npy'))
    depressed=newbg[:200].clone()
    for n,depth in zip(fill.HOLES,[2.,5.2,6.9]):
        affected=np.abs(fill.rowf-n)<=.25;depressed[:,:,affected]-=depth
    dump(OUT/'attention_rows.json',{str(n):np.where(np.abs(fill.rowf-n)<=.25)[0].tolist() for n in fill.HOLES})
    for tag,folder in [('Lb',LDEST),('M',DEST)]:
        for s in SEEDS:
            m,ck=getmodel(folder,s,'Lb')
            sets=[('filled_test',newtest,ti),('filled_bgA_test',newbg,[r for i,r in br]),('filled_notched_noise',newnn,[{}]*80)] if tag=='Lb' else [('old_bgA_test',oldbg,[r for i,r in br]),('old_notched_noise',oldnn,[{}]*80),('attention_before',newbg[:200],[r for i,r in br[:200]]),('attention_depressed',depressed,[r for i,r in br[:200]])]
            for name,x,rs in sets:
                probs=predict(m,x);ps={'argmax':probs.argmax(1),'thr_fa1':(probs[:,1]>=ck['thr_fa1']).astype(int)}
                writecsv(OUT/f'cross_{tag}_{name}_seed{s}.csv',[dict(r,p_drone=float(probs[i,1]),pred_argmax=int(ps['argmax'][i]),pred_thr=int(ps['thr_fa1'][i])) for i,r in enumerate(rs)])
                for mode,pred in ps.items():
                    ids=[i for i,r in enumerate(rs) if r.get('label','background')=='background'];k=int(pred[ids].sum());n=len(ids)
                    out.append(dict(model=tag,seed=s,input=name,mode=mode,k=k,n=n,rate=k/n,cp95_upper=cp(k,n)))
                    if name=='filled_test':
                        dr=[i for i,r in enumerate(rs) if r['label']=='drone'];k=int(pred[dr].sum());out.append(dict(model=tag,seed=s,input='filled_test_drone_recall',mode=mode,k=k,n=len(dr),rate=k/len(dr),cp95_upper=''))
            del m;torch.cuda.empty_cache()
    writecsv(OUT/'cross_metrics.csv',out);dump(OUT/'cross_done.json',dict(seconds=time.time()-start))
if __name__=='__main__':run()
