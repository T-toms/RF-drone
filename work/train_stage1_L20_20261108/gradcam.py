import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from evaluate import aggregate
def run():
    t=time.time();allstats=[];summary=[];features={r['group_id']:r for r in load(INPUT/'recording_features.json')}
    for b in BANDS:
        x,y,rs=dataset(b,'test');m,c=model(b,2026);acts=[]
        def hook(module,inputs,output):
            output.retain_grad();acts[:]=[output]
        h=m.layer4.register_forward_hook(hook);cams=[];preds=[]
        for i in range(0,len(x),16):
            m.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):z=m(x[i:i+16].cuda().float().contiguous(memory_format=torch.channels_last))
            pred=z.argmax(1);z.gather(1,pred[:,None]).float().sum().backward();a=acts[0].float();g=acts[0].grad.float();cam=(g.mean((2,3),keepdim=True)*a).sum(1).relu();cam=F.interpolate(cam[:,None],(256,256),mode='bilinear',align_corners=False)[:,0];cam=cam/(cam.amax((1,2),keepdim=True).clamp_min(1e-20));cams.append(cam.detach().cpu().numpy());preds.extend(pred.cpu().tolist())
        h.remove();cam=np.concatenate(cams);np.save(CACHE/f'{b}_gradcam.npy',cam.astype(np.float16))
        low,high=extent(b)[2:];freq=np.linspace(low,high,256,endpoint=False)+(high-low)/512
        for i,r in enumerate(rs):
            mask=np.zeros(256,dtype=bool)
            if r['label']=='drone':
                for sb in features[r['group_id']]['subbands']:mask|=(freq>=sb['lo'])&(freq<=sb['hi'])
                region='occupied_subbands'
            else:mask=((freq>=2401)&(freq<=2423))|((freq>=2426)&(freq<=2448));region='WiFi_2412_2437_pm11MHz'
            total=float(cam[i].sum());energy=float(cam[i,mask].sum()/total) if total>0 and mask.any() else float('nan')
            allstats.append(dict(band=b,sample_id=r['sample_id'],dataset=r['dataset'],label=r['label'],group_id=r['group_id'],region=region,region_frequency_fraction=float(mask.mean()),energy_fraction=energy,cam_sum=total,predicted_class=preds[i],correct=int(preds[i]==int(r['label']=='drone'))))
        groups=[('data','drone'),('v3','drone'),('v3','background')]
        fig,axs=plt.subplots(6,6,figsize=(25,20))
        for col,(ds,lab) in enumerate(groups):
            ids=[i for i,r in enumerate(rs) if r['dataset']==ds and r['label']==lab];correct=[i for i in ids if preds[i]==int(lab=='drone')];wrong=[i for i in ids if preds[i]!=int(lab=='drone')];rng=np.random.default_rng(2026);rng.shuffle(correct);rng.shuffle(wrong);chosen=wrong[:3]+correct[:3];chosen+=(wrong[3:]+correct[3:])[:6-len(chosen)]
            for row,i in enumerate(chosen):
                bounds=[v for sb in features.get(rs[i]['group_id'],{}).get('subbands',[]) for v in [sb['lo'],sb['hi']]] if lab=='drone' else []
                title=f'{ds} {lab} pred={preds[i]}\n{rs[i]["sample_id"]}'
                for ax in axs[row,col*2:col*2+2]:image(ax,x[i,0].numpy(),b,title,bounds)
                axs[row,col*2+1].imshow(cam[i],origin='lower',aspect='auto',extent=extent(b),cmap='jet',alpha=.5,vmin=0,vmax=1)
        fig.tight_layout();fig.savefig(OUT/f'fig4_gradcam_{b}.png',dpi=110);plt.close(fig)
        fig,axs=plt.subplots(1,4,figsize=(18,4))
        for ax,(ds,lab) in zip(axs,[('all','background'),('all','drone'),('data','drone'),('v3','drone')]):
            ids=[i for i,r in enumerate(rs) if r['label']==lab and (ds=='all' or r['dataset']==ds)];ax.imshow(cam[ids].mean(0),origin='lower',aspect='auto',extent=extent(b),vmin=0,vmax=1,cmap='jet');ax.set(title=f'{ds} {lab}, n={len(ids)}',xlabel='Time (ms)',ylabel='Frequency (MHz)')
            rr=[r for r in allstats if r['band']==b and r['label']==lab and (ds=='all' or r['dataset']==ds)];v=np.array([r['energy_fraction'] for r in rr]);good=np.isfinite(v);summary.append(dict(band=b,dataset=ds,label=lab,n=len(ids),valid_energy_n=int(good.sum()),zero_heat_n=sum(r['cam_sum']==0 for r in rr),energy_fraction_mean=float(v[good].mean()) if good.any() else float('nan'),energy_fraction_std=float(v[good].std(ddof=1)) if good.sum()>1 else float('nan')))
        fig.tight_layout();fig.savefig(OUT/f'fig4b_gradcam_mean_{b}.png',dpi=150);plt.close(fig);del m,cam,x;torch.cuda.empty_cache()
    writecsv(OUT/'gradcam_per_sample.csv',allstats);writecsv(OUT/'gradcam_summary.csv',summary);dump(OUT/'gradcam_done.json',dict(seconds=time.time()-t))
if __name__=='__main__':run()

