import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import *
import ctypes
CACHE=DEST
import argparse
p=argparse.ArgumentParser();p.add_argument('--band',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--epochs',type=int,default=40);a=p.parse_args()
assert a.band=='2.4GHz'
b=a.band;s=a.seed;tag=f'{b}_seed{s}'; smoke=a.epochs!=40
if not smoke and (CACHE/f'{tag}_done.json').exists():print('SKIP',tag);sys.exit(0)
seed(s);torch.backends.cudnn.benchmark=True
start=time.time();dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tx,ty,tr,tk=joined('train');vx,vy,vr,vk=joined('val')
# Entire float16 caches reside in host RAM; GPU batches use channels_last.
m=ResNet18(1,2).cuda().to(memory_format=torch.channels_last)
counts=torch.bincount(tk,minlength=3).float();w=torch.tensor([1.,.5,.5])/counts;w=w*(len(tx)/2);w=w.cuda()
tw=w[tk.cuda()];vw=w[vk.cuda()]
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=1e-5)
scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
best=(float('inf'),float('inf'));logs=[];torch.cuda.reset_peak_memory_stats()
first_epoch=1
lastpath=CACHE/f'{tag}_{"smoke_" if smoke else ""}last.pt'
if lastpath.exists():
    resume=torch.load(lastpath,map_location='cpu',weights_only=False)
    m.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);sched.load_state_dict(resume['scheduler'])
    scaler.load_state_dict(resume['scaler']);best=tuple(resume['best']);first_epoch=resume['epoch']+1
    logs=rows(OUT/f'train_log_{tag}{"_smoke" if smoke else ""}.csv')
    random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])

for epoch in range(first_epoch,a.epochs+1):
    t=time.time();m.train();order=torch.randperm(len(tx));ls=0.;den=0.
    for ids in order.split(64):
        x=tx[ids].cuda().float().contiguous(memory_format=torch.channels_last);y=ty[ids].cuda();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype):
            ce=F.cross_entropy(m(x),y,reduction='none');bw=tw[ids.cuda()];loss=(ce*bw).sum()/bw.sum()
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();ls+=float(loss.detach())*len(ids)
    sched.step();m.eval();vl=0.;vd=0.;ps=[]
    with torch.no_grad():
        for i in range(0,len(vx),64):
            x=vx[i:i+64].cuda().float().contiguous(memory_format=torch.channels_last);y=vy[i:i+64].cuda()
            with torch.autocast('cuda',dtype=dtype):logits=m(x)
            bw=vw[i:i+64];vl+=float((F.cross_entropy(logits.float(),y,reduction='none')*bw).sum());vd+=float(bw.sum());ps.append(logits.float().softmax(1).cpu().numpy())
    probs=np.concatenate(ps);pred=probs.argmax(1);yy=vy.numpy();fa=float((pred[vk.numpy()==1]==1).mean());fa_bgA=float((pred[vk.numpy()==2]==1).mean());miss=float((pred[yy==1]==0).mean());vl/=vd
    log=dict(epoch=epoch,train_loss=ls/len(tx),val_loss=vl,val_fa_v3=fa,val_fa_bgA=fa_bgA,val_miss=miss,balanced_accuracy=1-(fa+miss)/2,lr=opt.param_groups[0]['lr'],seconds=time.time()-t)
    logs.append(log);print(tag,log,flush=True)
    state=dict(model=m.state_dict(),epoch=epoch,val_fa_v3=fa,val_fa_bgA=fa_bgA,val_miss=miss,val_loss=vl,seed=s,band=b,sample_weights=w.cpu().tolist(),train_counts=counts.tolist(),amp=str(dtype),input='passband_power_pool_Hann1024_hop512_256x256')
    if (fa+fa_bgA+miss,vl)<best:
        best=(fa+fa_bgA+miss,vl)
        bg=np.sort(probs[vk.numpy()==1,1]);bga=np.sort(probs[vk.numpy()==2,1]);allowed=int(np.floor(.01*len(bg)));thr_v3=float(np.nextafter(bg[-1-allowed],np.float32(np.inf)));thr_bgA=float(np.nextafter(bga[-1-int(np.floor(.01*len(bga)))],np.float32(np.inf)));thr=max(thr_v3,thr_bgA)
        state.update(thr_fa1=thr,thr_v3=thr_v3,thr_bgA=thr_bgA,val_v3_background_n=len(bg),val_bgA_background_n=len(bga),val_background_n=len(bg)+len(bga),threshold_rule='p_drone >= nextafter(sorted_bg[-1-floor(.01*N)], +inf)',val_fa_thr=float((bg>=thr).mean()))
        torch.save(state,CACHE/f'{tag}_{"smoke_" if smoke else ""}best.pt')
    torch.save(dict(state,optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),best=best,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),CACHE/f'{tag}_{"smoke_" if smoke else ""}last.pt')
    writecsv(OUT/f'train_log_{tag}{"_smoke" if smoke else ""}.csv',logs)
class PMC(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
pm=PMC();pm.cb=ctypes.sizeof(pm);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(pm),pm.cb)
done=dict(peak_host_bytes=pm.PeakWorkingSetSize,band=b,seed=s,epochs=a.epochs,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),amp=str(dtype))
dump(CACHE/f'{tag}_{"smoke_" if smoke else ""}done.json',done)
if not smoke:
    with (OUT/'runs_done.txt').open('a',encoding='utf-8') as f:f.write(tag+'\n')

