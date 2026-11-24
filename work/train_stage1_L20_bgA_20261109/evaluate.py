import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from support import *
import math, functools
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
@functools.lru_cache(None)
def cp(k,n):
    if k==n:return 1.
    # Invert regularized beta(k+1,n-k), using the continued fraction (no scipy).
    a=k+1.;b=n-k
    def cf(a,b,x):
        qab=a+b;qap=a+1;qam=a-1;c=1.;d=1-qab*x/qap;d=1/max(d,1e-300);h=d
        for m in range(1,501):
            aa=m*(b-m)*x/((qam+2*m)*(a+2*m));d=1+aa*d;d=d if abs(d)>1e-300 else 1e-300;c=1+aa/c;c=c if abs(c)>1e-300 else 1e-300;d=1/d;h*=d*c
            aa=-(a+m)*(qab+m)*x/((a+2*m)*(qap+2*m));d=1+aa*d;d=d if abs(d)>1e-300 else 1e-300;c=1+aa/c;c=c if abs(c)>1e-300 else 1e-300;d=1/d;delta=d*c;h*=delta
            if abs(delta-1)<3e-14:break
        return h
    def beta(x):
        bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
        return bt*cf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*cf(b,a,1-x)/b
    lo=0.;hi=1.
    for _ in range(60):
        mid=(lo+hi)/2
        if beta(mid)<.95:lo=mid
        else:hi=mid
    return (lo+hi)/2
def tensor(a):return torch.from_numpy(np.array(a,dtype=np.float16)).unsqueeze(1)
def gset(r):return r['dataset']=='data' and r['model'] in ['evo lite 6k','evo ii pro v3']
MET=[]
def metric(round,s,mode,category,group,pred,ids,error=False,fa=False):
    if not len(ids):return
    n=len(ids);k=int((1-pred[ids] if error else pred[ids]).sum());MET.append(dict(round=round,seed=s,mode=mode,category=category,group=group,n=n,k=k,rate=k/n,fa=fa,cp95_upper=cp(k,n) if fa else ''))
def groupmetric(round,s,mode,category,rs,pred,keys,sel,fa=False,error=False):
    groups=collections.defaultdict(list)
    for i,r in enumerate(rs):
        if sel(r):groups[' | '.join(r.get(k,'') for k in keys)].append(i)
    for group,ids in groups.items():metric(round,s,mode,category,group,pred,ids,error=error,fa=fa)
def run():
    start=time.time();test=rows(OLD/'2.4GHz_test_spec_meta.csv');testx=tensor(np.load(OLD/'2.4GHz_test_spec.npy',mmap_mode='r'))
    reject=rows(OLD/'2.4GHz_rejected_test_spec_meta.csv');rejectx=tensor(np.load(OLD/'2.4GHz_rejected_test_spec.npy',mmap_mode='r'))
    # bgA keep is evaluated on receiver-A background, independently of source-drone keep.
    source={r['sample_id']:r for r in rows(INPUT/'index_L20.csv') if r['band']=='2.4GHz'}
    bga=np.load(BG/'bgA_spec.npy',mmap_mode='r');bgsets={}
    for split in ['test','val']:
        br=bgrows(split);rs=[]
        for i,r in br:
            r=dict(r);orig=source[r['paired_sample_id']]
            for k in ['model','state','bw_label']:r['source_'+k]=orig[k]
            rs.append(r)
        bgsets[split]=(rs,tensor(bga[[i for i,r in br]]))
    plan={r['block_file']:r for r in load(INPUT/'plan.json')};gs={r['group_id']:r['split'] for r in rows(INPUT/'group_split.csv')};qr=[];qx=[]
    for run in load(ROOT/'work/bgA_20261109/quiet013_runs.json'):
        r=plan[run['block']];z=np.load(run['block'],mmap_mode='r')
        for j in range(run['start_ms'],run['end_ms']-19,20):
            qr.append(dict(sample_id=f'quiet013_{len(qr)}',group_id=r['group_id'],split=gs[r['group_id']],training_label='drone' if gs[r['group_id']]=='train' else 'not_train',start_ms=j))
            qx.append(spec(np.array(z[j*61440:j*61440+1228800]),'2.4GHz'))
    assert len(qr)==19;writecsv(OUT/'quiet013_metadata.csv',qr);qx=tensor(qx)
    rng=np.random.default_rng(20261109);white=[]
    for i in range(80):
        z=(rng.standard_normal(1228800)+1j*rng.standard_normal(1228800)).astype(np.complex64);white.append(spec(z,'2.4GHz'))
    wx=tensor(white);nn=np.load(ROOT/'work/review_20261109/notched_noise_2.4GHz.npy',mmap_mode='r');nx=tensor([spec(np.array(z),'2.4GHz') for z in nn])
    MIX=[]
    for round,folder in [('H',OLD),('J',DEST)]:
        for s in SEEDS:
            m,ck=getmodel(folder,s)
            sets={'test':(test,testx),'rejected':(reject,rejectx),'bgA_test':bgsets['test'],'bgA_val':bgsets['val'],'quiet013':(qr,qx),'white_noise':([{}]*80,wx),'notched_noise':([{}]*80,nx)}
            for name,(rs,x) in sets.items():
                probs=predict(m,x);pd=probs[:,1];preds={'argmax':probs.argmax(1),'thr_fa1':(pd>=ck['thr_fa1']).astype(int)}
                out=[dict(r,p_drone=float(pd[i]),pred_argmax=int(preds['argmax'][i]),pred_thr=int(preds['thr_fa1'][i])) for i,r in enumerate(rs)]
                filename=f'test_predictions_2.4GHz_seed{s}.csv' if name=='test' else f'{name}_predictions_seed{s}.csv'
                writecsv(OUT/((('H_' if round=='H' else '')+filename)),out)
                for mode,pred in preds.items():
                    if name=='test':
                        bg=[i for i,r in enumerate(rs) if r['label']=='background'];dr=[i for i,r in enumerate(rs) if r['label']=='drone'];ng=[i for i in dr if not gset(rs[i])]
                        metric(round,s,mode,'main','v3_test_FA',pred,bg,fa=True)
                        metric(round,s,mode,'main','nonG_miss',pred,ng,error=True);metric(round,s,mode,'main','all_drone_miss_with_G',pred,dr,error=True)
                        groupmetric(round,s,mode,'fa_by_session',rs,pred,['group_id'],lambda r:r['label']=='background',fa=True)
                        groupmetric(round,s,mode,'miss_by_model_state_bw',rs,pred,['dataset','model','state','bw_label'],lambda r:r['label']=='drone',error=True)
                        groupmetric(round,s,mode,'G_recall',rs,pred,['model'],gset)
                        groupmetric(round,s,mode,'source_recall',rs,pred,['dataset'],lambda r:r['label']=='drone')
                        groupmetric(round,s,mode,'same_model_recall',rs,pred,['dataset','model','bw_label'],lambda r:r['label']=='drone' and r['model'] in ['mini5 pro','mini5pro','evo ii pro v3','evo2pro_v3'])
                        metric(round,s,mode,'main','data_nonG_recall',pred,[i for i in ng if rs[i]['dataset']=='data'])
                        metric(round,s,mode,'main','v3_drone_recall',pred,[i for i in dr if rs[i]['dataset']=='v3'])
                    elif name=='rejected':
                        metric(round,s,mode,'rejected','all_rejected_recall',pred,list(range(len(rs))))
                        groupmetric(round,s,mode,'rejected_detail',rs,pred,['dataset','model','state','bw_label'],lambda r:True)
                    else:
                        metric(round,s,mode,'main',name+'_FA',pred,list(range(len(rs))),fa=True)
                        if name.startswith('bgA'):groupmetric(round,s,mode,name+'_by_source',rs,pred,['source_model','source_state','source_bw_label'],lambda r:True,fa=True)
                        if name=='quiet013':groupmetric(round,s,mode,'quiet013_by_group_split',rs,pred,['group_id','split','training_label'],lambda r:True,fa=True)
            # Read-only mix caches; stage in float16 one X at a time.
            for group in ['data_drone','v3_drone','v3_background']:
                arr=np.load(OLD/f'2.4GHz_mix_{group}.npy',mmap_mode='r')
                for k,db in enumerate([-10,-5,0,5,10,15,20]):
                    probs=predict(m,tensor(arr[:,k]))
                    for mode,pred in [('argmax',probs.argmax(1)),('thr_fa1',(probs[:,1]>=ck['thr_fa1']).astype(int))]:
                        n=len(pred);hits=int(pred.sum());MIX.append(dict(round=round,seed=s,mode=mode,group=group,x_db=db,n=n,k=hits,rate=hits/n,cp95_upper=cp(hits,n) if group=='v3_background' else ''))
            del m;torch.cuda.empty_cache();print('EVALUATED',round,s,flush=True)
            writecsv(OUT/'metrics_all.csv',MET);writecsv(OUT/'mix_results.csv',MIX)
    dump(OUT/'evaluation_done.json',dict(seconds=time.time()-start))
if __name__=='__main__':run()
