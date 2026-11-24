import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--band',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--epochs',type=int,default=40);a=p.parse_args()
b=a.band;s=a.seed;tag=f'{b}_seed{s}'; smoke=a.epochs!=40
if not smoke and (OUT/f'{tag}_done.json').exists():print('SKIP',tag);sys.exit(0)
seed(s);torch.backends.cudnn.benchmark=True
start=time.time();dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tx,ty,tr=dataset(b,'train');vx,vy,vr=dataset(b,'val')
# Entire float16 caches reside in host RAM; GPU batches use channels_last.
m=ResNet18(1,2).cuda().to(memory_format=torch.channels_last)
counts=torch.bincount(ty,minlength=2).float();w=1/counts;w=w/w.mean();w=w.cuda()
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=1e-5)
scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
best=(float('inf'),float('inf'));logs=[];torch.cuda.reset_peak_memory_stats()
for epoch in range(1,a.epochs+1):
    t=time.time();m.train();order=torch.randperm(len(tx));ls=0.;den=0.
    for ids in order.split(64):
        x=tx[ids].cuda().float().contiguous(memory_format=torch.channels_last);y=ty[ids].cuda();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype):loss=F.cross_entropy(m(x),y,weight=w)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();ls+=float(loss.detach())*len(ids)
    sched.step();m.eval();vl=0.;vd=0.;ps=[]
    with torch.no_grad():
        for i in range(0,len(vx),64):
            x=vx[i:i+64].cuda().float().contiguous(memory_format=torch.channels_last);y=vy[i:i+64].cuda()
            with torch.autocast('cuda',dtype=dtype):logits=m(x)
            vl+=float(F.cross_entropy(logits.float(),y,weight=w,reduction='sum'));vd+=float(w[y].sum());ps.append(logits.float().softmax(1).cpu().numpy())
    probs=np.concatenate(ps);pred=probs.argmax(1);yy=vy.numpy();fa=float((pred[yy==0]==1).mean());miss=float((pred[yy==1]==0).mean());vl/=vd
    log=dict(epoch=epoch,train_loss=ls/len(tx),val_loss=vl,val_fa=fa,val_miss=miss,balanced_accuracy=1-(fa+miss)/2,lr=opt.param_groups[0]['lr'],seconds=time.time()-t)
    logs.append(log);print(tag,log,flush=True)
    state=dict(model=m.state_dict(),epoch=epoch,val_fa=fa,val_miss=miss,val_loss=vl,seed=s,band=b,class_weights=w.cpu().tolist(),amp=str(dtype),input='passband_power_pool_Hann1024_hop512_256x256')
    if (fa+miss,vl)<best:
        best=(fa+miss,vl)
        bg=np.sort(probs[yy==0,1]);allowed=int(np.floor(.01*len(bg)));thr=float(np.nextafter(bg[-1-allowed],np.float32(np.inf)))
        state.update(thr_fa1=thr,val_background_n=len(bg),threshold_rule='p_drone >= nextafter(sorted_bg[-1-floor(.01*N)], +inf)',val_fa_thr=float((bg>=thr).mean()))
        torch.save(state,CACHE/f'{tag}_{"smoke_" if smoke else ""}best.pt')
    torch.save(dict(state,optimizer=opt.state_dict(),scheduler=sched.state_dict()),CACHE/f'{tag}_{"smoke_" if smoke else ""}last.pt')
    writecsv(OUT/f'train_log_{tag}{"_smoke" if smoke else ""}.csv',logs)
done=dict(band=b,seed=s,epochs=a.epochs,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),amp=str(dtype))
dump(OUT/f'{tag}_{"smoke_" if smoke else ""}done.json',done)
if not smoke:
    with (OUT/'runs_done.txt').open('a',encoding='utf-8') as f:f.write(tag+'\n')

