import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from evaluate import aggregate
XS=[-10,-5,0,5,10,15,20]
def run():
    t=time.time();results=[];pairs=[]
    for b in BANDS:
        rs=meta(b,'test');bg=[r for r in rs if r['dataset']=='v3' and r['label']=='background'];background=[];old=None
        for r in bg:
            if r['block_file']!=old:block=np.load(r['block_file'],mmap_mode='r');old=r['block_file']
            lo=int(r['start']);background.append(np.array(block[lo:lo+int(r['n_samples'])]))
        powers=np.array([np.mean(np.abs(z)**2,dtype=np.float64) for z in background]);models=[model(b,s) for s in SEEDS]
        for group in ['data_drone','v3_drone','v3_background']:
            ds,lab=group.split('_');source=[r for r in rs if r['dataset']==ds and r['label']==('background' if lab=='background' else 'drone')]
            path=CACHE/f'{b}_mix_{group}.npy';done=path.with_suffix('.done.json')
            if not done.exists():
                a=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(source),len(XS),256,256));old=None
                for i,r in enumerate(source):
                    if r['block_file']!=old:block=np.load(r['block_file'],mmap_mode='r');old=r['block_file']
                    lo=int(r['start']);d=np.array(block[lo:lo+int(r['n_samples'])]);pd=np.mean(np.abs(d)**2,dtype=np.float64);j=(i+1)%len(bg) if lab=='background' else i%len(bg);base=background[j]
                    for k,x in enumerate(XS):
                        g=float(np.sqrt(powers[j]/max(pd,1e-30)*10**(x/10)));a[i,k]=spec(base+g*d,b)
                        pairs.append(dict(band=b,group=group,sample_id=r['sample_id'],background_sample_id=bg[j]['sample_id'],x_db=x,g=g,source_power=float(pd),background_power=float(powers[j]),achieved_ratio_db=float(10*np.log10(g*g*pd/powers[j]))))
                    if i%100==0:print(b,group,i,len(source),flush=True)
                    if i==0 and group=='data_drone':
                        fig,axs=plt.subplots(1,4,figsize=(18,4));image(axs[0],spec(base,b),b,'v3 background')
                        for ax,x in zip(axs[1:],[-5,5,15]):image(ax,a[i,XS.index(x)],b,f'data drone + v3 background: {x} dB')
                        fig.tight_layout();fig.savefig(OUT/f'fig3b_mix_examples_{b}.png',dpi=150);plt.close(fig)
                a.flush();writecsv(path.with_suffix('.pairs.csv'),[r for r in pairs if r['band']==b and r['group']==group]);dump(done,dict(n=len(source),shape=list(a.shape)));del a
            elif path.with_suffix('.pairs.csv').exists():pairs.extend(rows(path.with_suffix('.pairs.csv')))
            arr=np.load(path)
            for s,(m,c) in zip(SEEDS,models):
                for k,x in enumerate(XS):
                    probs=predict(m,torch.from_numpy(arr[:,k].copy()).unsqueeze(1))
                    for mode,pred in [('argmax',probs.argmax(1)),('thr_fa1',(probs[:,1]>=c['thr_fa1']).astype(int))]:results.append(dict(band=b,group=group,seed=s,mode=mode,x_db=x,n=len(source),detection_rate=float(pred.mean())))
            del arr
        del models,background;torch.cuda.empty_cache()
    writecsv(OUT/'mix_results.csv',results);writecsv(OUT/'mix_pairs.csv',pairs);summary=aggregate(results,['band','group','mode','x_db','n'],['detection_rate']);writecsv(OUT/'mix_summary.csv',summary)
    for b in BANDS:
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for ax,mode in zip(axs,['argmax','thr_fa1']):
            for group in ['data_drone','v3_drone','v3_background']:
                rr=[r for r in summary if r['band']==b and r['mode']==mode and r['group']==group];ax.errorbar([r['x_db'] for r in rr],[r['detection_rate_mean'] for r in rr],yerr=[r['detection_rate_std'] for r in rr],marker='o',label=group+(' FA' if group=='v3_background' else ' recall'))
            ax.set(xlabel='Added component / background power (dB)',ylabel='Detection / false alarm rate',title=mode,ylim=(-.03,1.03));ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(OUT/f'fig3_mix_test_{b}.png',dpi=150);plt.close(fig)
    dump(OUT/'mix_done.json',dict(seconds=time.time()-t))
if __name__=='__main__':run()

