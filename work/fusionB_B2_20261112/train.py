from base import *
sys.path.insert(0,str(ROOT/'work/train_stage1_L20_20261108'))
from common import seed,ResNet18,torch,nn,F
NAMES=['shape_factor','log_kurt_a','papr999_db','c42n','log_gamma_max','sigma_f','spec_flatness','occ_frac','duty','log1p_n_bursts']
F5COLS=[2,5,7,8,9]
def feature_names(g):return [NAMES[i] for i in F5COLS] if g=='F5' else NAMES
class Fusion(nn.Module):
 def __init__(self,n_features=10):
  super().__init__();r=ResNet18(1,2)
  for k in ['stem','layer1','layer2','layer3','layer4','pool']:setattr(self,k,getattr(r,k))
  self.image_norm=nn.LayerNorm(512);self.features=nn.Sequential(nn.Linear(n_features,64),nn.ReLU(),nn.LayerNorm(64));self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(576,2))
 def forward(self,x,f):
  for k in ['stem','layer1','layer2','layer3','layer4','pool']:x=getattr(self,k)(x)
  return self.head(torch.cat([self.image_norm(x.flatten(1)),self.features(f)],1))
def loadset(name):return np.load(CACHE/(name+'_spec.npy'),mmap_mode='r'),np.load(CACHE/(name+'_features.npy')),rows(CACHE/(name+'_meta.csv'))
def joined(b,s):
 x,f,r=loadset(b+'_'+s);xx=[x];ff=[f];k=[0 if z['label']=='drone' else 1 for z in r]
 if b=='2.4GHz':
  ax,af,ar=loadset('bgA');ii=[i for i,z in enumerate(ar) if z['split']==s];xx.append(ax[ii]);ff.append(af[ii]);r+= [ar[i] for i in ii];k +=[2]*len(ii)
  if s=='train':
   for name,kind in [('drone',3),('bg',4)]:
    ax,af,ar=loadset(b+'_mix_train_'+name);xx.append(ax);ff.append(af);r+=ar;k +=[kind]*len(ar)
 return torch.from_numpy(np.concatenate(xx)).unsqueeze(1),torch.tensor([z['label']=='drone' for z in r],dtype=torch.long),np.concatenate(ff),r,torch.tensor(k)
def stats(b,f):
 med=np.nanmedian(f,axis=0);med=np.where(np.isfinite(med),med,0);z=np.where(np.isfinite(f),f,med);mean=z.mean(0);std=z.std(0);std=np.where(std>0,std,1);v=dict(names=NAMES,median=med.tolist(),mean=mean.tolist(),std=std.tolist(),train_missing=(~np.isfinite(f)).sum(0).tolist(),train_clip=(abs((z-mean)/std)>10).sum(0).tolist());dump(OUT/(b+'_normalization.json'),v);return v
def transform(f,v):return np.clip((np.where(np.isfinite(f),f,np.array(v['median']))-np.array(v['mean']))/np.array(v['std']),-10,10).astype('float32')
def forward(m,x,f,g):return m(x) if g=='R' else m(x,f[:,F5COLS] if g=='F5' else f if g=='F' else torch.zeros_like(f))
def train(b,g,s,epochs=40):
 import random,ctypes
 tag=f'{g}_{b}_seed{s}'+('_smoke' if epochs!=40 else '');done=OUT/(tag+'_done.json')
 if done.exists():return
 seed(s);torch.backends.cudnn.benchmark=True;start=time.time();dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 tx,ty,tf,tr,tk=joined(b,'train');vx,vy,vf,vr,vk=joined(b,'val');norm=stats(b,tf);tf=torch.from_numpy(transform(tf,norm));vf=torch.from_numpy(transform(vf,norm))
 m=(ResNet18(1,2) if g=='R' else Fusion(5 if g=='F5' else 10)).cuda().to(memory_format=torch.channels_last)
 if b=='2.4GHz':
  counts=torch.bincount(tk,minlength=5).float();w=torch.tensor([.25,.25,.125,.25,.125])/counts*len(tx);tw=w[tk].cuda();jcounts=counts[:3];vw=(torch.tensor([.5,.25,.25])/jcounts*jcounts.sum())[vk].cuda()
 else:
  counts=torch.bincount(ty,minlength=2).float();w=1/counts;w=w/w.mean();tw=w[ty].cuda();vw=w[vy].cuda()
 opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=1e-5);scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
 best=(float('inf'),float('inf'));logs=[];first=1;last=CACHE/(tag+'_last.pt');torch.cuda.reset_peak_memory_stats()
 if last.exists():
  ck=torch.load(last,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler']);best=tuple(ck['best']);first=ck['epoch']+1;logs=rows(OUT/(tag+'_epochs.csv'));random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 for epoch in range(first,epochs+1):
  t=time.time();m.train();order=torch.randperm(len(tx));ls=0
  for ids in order.split(64):
   x=tx[ids].cuda().float().contiguous(memory_format=torch.channels_last);y=ty[ids].cuda();f=tf[ids].cuda();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=dtype):
    ce=F.cross_entropy(forward(m,x,f,g),y,reduction='none');bw=tw[ids.cuda()];loss=(ce*bw).sum()/bw.sum()
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();ls+=float(loss.detach())*len(ids)
  sched.step();m.eval();vl=0.;vd=0.;ps=[]
  with torch.no_grad():
   for i in range(0,len(vx),64):
    x=vx[i:i+64].cuda().float().contiguous(memory_format=torch.channels_last);y=vy[i:i+64].cuda();f=vf[i:i+64].cuda()
    with torch.autocast('cuda',dtype=dtype):logits=forward(m,x,f,g)
    bw=vw[i:i+64];vl+=float((F.cross_entropy(logits.float(),y,reduction='none')*bw).sum());vd+=float(bw.sum());ps.append(logits.float().softmax(1).cpu().numpy())
  probs=np.concatenate(ps);pred=probs.argmax(1);yy=vy.numpy();fa=float((pred[vk.numpy()==1]==1).mean());miss=float((pred[yy==1]==0).mean());faA=float((pred[vk.numpy()==2]==1).mean()) if b=='2.4GHz' else 0.;vl/=vd
  log=dict(epoch=epoch,train_loss=ls/len(tx),val_loss=vl,val_fa_v3=fa,val_fa_bgA=faA,val_miss=miss,balanced_accuracy=1-(fa+miss)/2,lr=opt.param_groups[0]['lr'],seconds=time.time()-t);
  if b=='5.8GHz':log['val_fa']=fa
  logs.append(log);print(tag,log,flush=True)
  state=dict(model=m.state_dict(),epoch=epoch,band=b,seed=s,group=g,normalization=norm,feature_names=feature_names(g),val_fa_v3=fa,val_fa_bgA=faA,val_miss=miss,val_loss=vl,amp=str(dtype),counts=counts.tolist(),weights=w.tolist(),parameters=sum(p.numel() for p in m.parameters()))
  if b=='5.8GHz':state['val_fa']=fa
  if (fa+faA+miss,vl)<best:
   best=(fa+faA+miss,vl);thresholds=[]
   for kind in ([1,2] if b=='2.4GHz' else [1]):
    bg=np.sort(probs[vk.numpy()==kind,1]);thresholds.append(float(np.nextafter(bg[-1-int(np.floor(.01*len(bg)))],np.float32(np.inf))))
   state.update(thr_fa1=max(thresholds),background_thresholds=thresholds);torch.save(state,CACHE/(tag+'_best.pt'))
  torch.save(dict(state,optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),best=best,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),last);writecsv(OUT/(tag+'_epochs.csv'),logs)
 class PMC(ctypes.Structure):
  _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
 pm=PMC();pm.cb=ctypes.sizeof(pm);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(pm),pm.cb)
 dump(done,dict(seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_host_bytes=pm.PeakWorkingSetSize,epochs=epochs,tag=tag));del m;torch.cuda.empty_cache()
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--band',required=True);p.add_argument('--group',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=40);a=p.parse_args();train(a.band,a.group,a.seed,a.epochs)
