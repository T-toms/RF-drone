import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def aggregate(rs,keys,values):
    out=[];groups=collections.defaultdict(list)
    for r in rs:groups[tuple(r[k] for k in keys)].append(r)
    for k,rr in groups.items():
        z=dict(zip(keys,k))
        for v in values:
            vals=np.array([float(r[v]) for r in rr]);z[v+'_mean']=float(vals.mean());z[v+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else 0.
        out.append(z)
    return out
def run():
    t=time.time();src=[];same=[];detail=[];bgdetail=[];main=[];rejects=[];strength=[];conf=[]
    # One sequential traversal per block for test power proxies and rejected caches.
    index=rows(INPUT/'index_L20.csv')
    for b in BANDS:
        rs=[dict(r) for r in index if r['band']==b and r['split']=='test'];rs.sort(key=lambda r:(r['block_file'],int(r['start'])))
        rej=[r for r in rs if r['keep']!='True' and r['label']=='drone'];ra=np.empty((len(rej),256,256),dtype=np.float16);ri=0;old=None
        for r in rs:
            if r['block_file']!=old:block=np.load(r['block_file'],mmap_mode='r');old=r['block_file']
            lo=int(r['start']);iq=np.array(block[lo:lo+int(r['n_samples'])]);pw=np.mean(np.abs(iq.reshape(20,FS[b]//1000))**2,axis=1);db=10*np.log10(np.maximum(pw,1e-30));q10,q90=np.quantile(db,[.1,.9]);r['power_proxy_db']=float(q90-q10)
            strength.append(dict(band=b,sample_id=r['sample_id'],dataset=r['dataset'],model=r['model'],label=r['label'],keep=r['keep'],power_p10_db=float(q10),power_p90_db=float(q90),power_proxy_db=float(q90-q10)))
            if r['keep']!='True' and r['label']=='drone':ra[ri]=spec(iq,b);ri+=1
        np.save(CACHE/f'{b}_rejected_test_spec.npy',ra);writecsv(CACHE/f'{b}_rejected_test_spec_meta.csv',rej)
        x,y,meta_rows=dataset(b,'test');yy=y.numpy()
        for s in SEEDS:
            m,c=model(b,s);probs=predict(m,x);pd=probs[:,1];preds={'argmax':probs.argmax(1),'thr_fa1':(pd>=c['thr_fa1']).astype(int)}
            out=[]
            for i,r in enumerate(meta_rows):out.append(dict(r,p_drone=float(pd[i]),pred_argmax=int(preds['argmax'][i]),pred_thr=int(preds['thr_fa1'][i])))
            writecsv(OUT/f'test_predictions_{b}_seed{s}.csv',out)
            for mode,pred in preds.items():
                main.append(dict(band=b,seed=s,mode=mode,n_background=int((yy==0).sum()),n_drone=int((yy==1).sum()),fa=float(pred[yy==0].mean()),miss=float((pred[yy==1]==0).mean()),threshold=.5 if mode=='argmax' else c['thr_fa1']))
                matrix=np.bincount(yy*2+pred,minlength=4).reshape(2,2);conf.append(dict(band=b,seed=s,mode=mode,TN=int(matrix[0,0]),FP=int(matrix[0,1]),FN=int(matrix[1,0]),TP=int(matrix[1,1])))
                for dataset_name in ['data','v3']:
                    sel=np.array([r['dataset']==dataset_name and r['label']=='drone' for r in meta_rows]);src.append(dict(band=b,seed=s,mode=mode,dataset=dataset_name,n=int(sel.sum()),recall=float(pred[sel].mean())))
                groups=collections.defaultdict(list)
                for i,r in enumerate(meta_rows):groups[(r['dataset'],r['model'],r['state'],r['bw_label'])].append(i)
                for (ds,md,st,bw),ids in groups.items():
                    if md=='none':continue
                    z=dict(band=b,seed=s,mode=mode,dataset=ds,model=md,state=st,bw_label=bw,n=len(ids),miss=float((pred[ids]==0).mean()));detail.append(z)
                groups=collections.defaultdict(list)
                for i,r in enumerate(meta_rows):
                    if r['label']=='background':groups[r['group_id']].append(i)
                for gid,ids in groups.items():bgdetail.append(dict(band=b,seed=s,mode=mode,group_id=gid,n=len(ids),fa=float(pred[ids].mean())))
                for canonical,aliases in [('Mini 5 Pro',{'mini5 pro','mini5pro'}),('EVO II Pro V3',{'evo ii pro v3','evo2pro_v3'})]:
                    groups=collections.defaultdict(list)
                    for i,r in enumerate(meta_rows):
                        if r['model'] in aliases:groups[(r['dataset'],r['bw_label'])].append(i)
                    for (ds,bw),ids in groups.items():same.append(dict(band=b,seed=s,mode=mode,canonical_model=canonical,dataset=ds,bw_label=bw,n=len(ids),recall=float(pred[ids].mean())))
            if len(rej):
                rp=predict(m,torch.from_numpy(ra).unsqueeze(1))
                for mode,rr in [('argmax',rp.argmax(1)),('thr_fa1',(rp[:,1]>=c['thr_fa1']).astype(int))]:rejects.append(dict(band=b,seed=s,mode=mode,n=len(rej),recall=float(rr.mean())))
            del m;torch.cuda.empty_cache()
        fig,axs=plt.subplots(1,3,figsize=(15,4))
        for s in SEEDS:
            logs=rows(OUT/f'train_log_{b}_seed{s}.csv')
            for ax,key in zip(axs,['val_fa','val_miss','val_loss']):ax.plot([int(r['epoch']) for r in logs],[float(r[key]) for r in logs],label=str(s));ax.set(xlabel='Epoch',ylabel=key);ax.legend()
        fig.tight_layout();fig.savefig(OUT/f'fig5_curves_{b}.png',dpi=150);plt.close(fig)
    for name,rs in [('main_metrics',main),('recall_by_source',src),('recall_same_model',same),('miss_by_model_state_bw',detail),('fa_by_session',bgdetail),('rejected_test_recall',rejects),('test_power_proxy',strength),('confusion_matrices',conf)]:writecsv(OUT/f'{name}.csv',rs)
    for name,rs,keys,vals in [('main_metrics',main,['band','mode'],['fa','miss']),('recall_by_source',src,['band','mode','dataset','n'],['recall']),('recall_same_model',same,['band','mode','canonical_model','dataset','bw_label','n'],['recall']),('miss_by_model_state_bw',detail,['band','mode','dataset','model','state','bw_label','n'],['miss']),('fa_by_session',bgdetail,['band','mode','group_id','n'],['fa']),('rejected_test_recall',rejects,['band','mode','n'],['recall'])]:writecsv(OUT/f'{name}_summary.csv',aggregate(rs,keys,vals))
    for b in BANDS:
        fig,axs=plt.subplots(1,2,figsize=(14,5));rr=[r for r in aggregate(src,['band','mode','dataset'],['recall']) if r['band']==b and r['mode']=='argmax'];ss=[r for r in aggregate(same,['band','mode','canonical_model','dataset'],['recall']) if r['band']==b and r['mode']=='argmax'];allr=rr+ss;labels=[r.get('canonical_model','All')+' '+r['dataset'] for r in allr];axs[0].bar(range(len(allr)),[r['recall_mean'] for r in allr],yerr=[r['recall_std'] for r in allr]);axs[0].set_xticks(range(len(allr)),labels,rotation=35,ha='right');axs[0].set(ylabel='Recall (argmax)',ylim=(0,1.05))
        for ds in ['data','v3']:
            a=[r['power_proxy_db'] for r in strength if r['band']==b and r['dataset']==ds and r['label']=='drone' and r['keep']=='True'];axs[1].hist(a,bins=25,alpha=.5,label=ds)
        axs[1].set(xlabel='1 ms power P90 - P10 (dB)',ylabel='Test drone count');axs[1].legend();fig.tight_layout();fig.savefig(OUT/f'fig2_recall_by_source_{b}.png',dpi=150);plt.close(fig)
    dump(OUT/'evaluate_done.json',dict(seconds=time.time()-t))
if __name__=='__main__':run()
