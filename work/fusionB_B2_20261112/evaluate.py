from train import *
from collections import defaultdict
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
def isG(r,b):return (r.get('dataset')=='data' and r.get('model') in ['evo lite 6k','evo ii pro v3']) if b=='2.4GHz' else (r.get('dataset')=='v3' and r.get('model') in ['evo2pro_v3','fpv'])
@torch.no_grad()
def predict(m,x,f,g):
 if g in ['F','F5']:
  # Evaluation-only reuse of the image branch, which is independent of IQ features.
  cache=getattr(m,'_image_cache',{})
  if id(x) not in cache:
   zz=[];dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
   for i in range(0,len(x),64):
    z=torch.from_numpy(np.array(x[i:i+64])).unsqueeze(1).cuda().float().contiguous(memory_format=torch.channels_last)
    with torch.autocast('cuda',dtype=dtype):
     for k in ['stem','layer1','layer2','layer3','layer4','pool']:z=getattr(m,k)(z)
     z=m.image_norm(z.flatten(1))
    zz.append(z.cpu())
   cache[id(x)]=torch.cat(zz);m._image_cache=cache
  pp=[]
  for i in range(0,len(x),64):
   ff=torch.from_numpy(f[i:i+64]).cuda();ff=ff[:,F5COLS] if g=='F5' else ff
   with torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):p=m.head(torch.cat([cache[id(x)][i:i+64].cuda(),m.features(ff)],1))
   pp.append(p.float().softmax(1).cpu().numpy()[:,1])
  result=np.concatenate(pp)
  if not getattr(m,'_cache_validated',False):
   n=min(64,len(x));z=torch.from_numpy(np.array(x[:n])).unsqueeze(1).cuda().float().contiguous(memory_format=torch.channels_last)
   with torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):direct=forward(m,z,torch.from_numpy(f[:n]).cuda(),g)
   delta=float(np.max(abs(result[:n]-direct.float().softmax(1).cpu().numpy()[:,1])))
   assert delta<=1e-6,('image branch reuse',delta)
   m._cache_validated=True;print('IMAGE_CACHE_VALIDATED',g,delta,flush=True)
  return result
 pp=[];dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 for i in range(0,len(x),64):
  z=torch.from_numpy(np.array(x[i:i+64])).unsqueeze(1).cuda().float().contiguous(memory_format=torch.channels_last);ff=torch.from_numpy(f[i:i+64]).cuda()
  with torch.autocast('cuda',dtype=dtype):p=forward(m,z,ff,g)
  pp.append(p.float().softmax(1).cpu().numpy()[:,1])
 return np.concatenate(pp)
def metrics(b,g,s,name,rs,p,thr,dependence='original',repeat=0):
 out=[]
 for mode,pred in [('argmax',(p>.5).astype(int)),('thr_fa1',(p>=thr).astype(int))]:
  def add(category,group,ids,error=False,fa=False):
   if not len(ids):return
   n=len(ids);k=int((1-pred[ids] if error else pred[ids]).sum());out.append(dict(band=b,group=g,seed=s,set=name,mode=mode,category=category,subset=group,n=n,k=k,rate=k/n,cp95_upper=(float(beta.ppf(.95,k+1,n-k)) if k<n else 1.) if fa else '',dependence=dependence,repeat=repeat))
  def grouped(cat,keys,sel,error=False,fa=False):
   groups=defaultdict(list)
   for i,r in enumerate(rs):
    if sel(r):groups[' | '.join(str(r.get(k,'')) for k in keys)].append(i)
   for key,ii in groups.items():add(cat,key,ii,error,fa)
  if name=='test':
   dr=[i for i,r in enumerate(rs) if r['label']=='drone'];ng=[i for i in dr if not isG(rs[i],b)];bg=[i for i,r in enumerate(rs) if r['label']=='background' and r['dataset']=='v3']
   add('main','v3_test_FA',bg,fa=True)
   if any(r.get('dataset')=='dataA_mirror' for r in rs):add('main','bgA_test_FA',[i for i,r in enumerate(rs) if r.get('dataset')=='dataA_mirror'],fa=True)
   add('main','nonG_miss',ng,error=True);add('main','all_drone_miss_with_G',dr,error=True);add('main','v3_drone_recall',[i for i in dr if rs[i]['dataset']=='v3']);add('main','data_nonG_recall',[i for i in ng if rs[i]['dataset']=='data'])
   add('receiver','B_FA',bg,fa=True);add('receiver','B_nonG_recall',[i for i in ng if rs[i]['dataset']=='v3']);add('receiver','A_nonG_recall',[i for i in ng if rs[i]['dataset']=='data'])
   grouped('fa_by_session',['group_id'],lambda r:r['label']=='background',fa=True);grouped('miss_by_model_state_bw',['dataset','model','state','bw_label'],lambda r:r['label']=='drone',error=True);grouped('G_recall',['dataset','model'],lambda r:isG(r,b));grouped('source_recall',['dataset'],lambda r:r['label']=='drone');grouped('v3_model_recall',['model'],lambda r:r['label']=='drone' and r['dataset']=='v3');grouped('same_model_recall',['dataset','model','bw_label'],lambda r:r['label']=='drone' and r['model'] in ['mini5 pro','mini5pro','evo ii pro v3','evo2pro_v3'])
   for yy in [0,1]:
    ids=[i for i,r in enumerate(rs) if int(r['label']=='drone')==yy]
    for zz in [0,1]:add('confusion',f'{yy}->{zz}',ids,error=zz==0)
  elif name=='rejected_test':
   add('rejected','all_rejected_recall',list(range(len(rs))));grouped('rejected_detail',['dataset','model','state','bw_label'],lambda r:True)
  elif name=='mix_test':
   grouped('mix_recall',['group','x_db'],lambda r:r['group']!='v3_background');grouped('mix_FA',['group','x_db'],lambda r:r['group']=='v3_background',fa=True);grouped('mix_model',['group','model','x_db'],lambda r:r['group']!='v3_background')
  else:
   add('main',name+'_FA',list(range(len(rs))),fa=True)
   if name=='bgA_test':add('receiver','A_FA',list(range(len(rs))),fa=True)
   if name in ['C1','quiet013','white_noise','notched_noise']:add('receiver',name+'_FA',list(range(len(rs))),fa=True)
   if name=='C1':grouped('C1',['x_db'],lambda r:True,fa=True)
   if name.startswith('bgA'):grouped('bgA_by_source',['source_model','source_state','source_bw_label'],lambda r:True,fa=True)
   if name=='quiet013':grouped('quiet013_by_group',['group_id','split','training_label'],lambda r:True,fa=True)
 return out
def sets(b):
 result={s:loadset(b+'_'+s) for s in ['test','val','rejected_test','mix_test','white_noise','notched_noise']}
 result.pop('val')
 if b=='2.4GHz':
  result.update({s:loadset(b+'_'+s) for s in ['quiet013','C1']});x,f,rs=loadset('bgA');source={r['sample_id']:r for r in rows(ROOT/'work/crop_slice_20261107/index_L20.csv')}
  quiet={r['sample_id']:r for r in rows(ROOT/'work/side_L_wifimix_20261110/quiet013_metadata.csv')}
  for r in result['quiet013'][2]:r.update(quiet[r['sample_id']])
  for split in ['test','val']:
   ii=[i for i,r in enumerate(rs) if r['split']==split];rr=[]
   for i in ii:
    r=dict(rs[i]);src=source[r['paired_sample_id']]
    for k in ['model','state','bw_label']:r['source_'+k]=src[k]
    rr.append(r)
   result['bgA_'+split]=(x[ii],f[ii],rr)
 return result
def aggregate(rr,keys,values=['rate']):
 groups=defaultdict(list)
 for r in rr:groups[tuple(r[k] for k in keys)].append(r)
 out=[]
 for kk,rs in groups.items():
  z=dict(zip(keys,kk))
  for v in values:
   vals=np.array([float(r[v]) for r in rs]);z[v+'_mean']=float(vals.mean());z[v+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else 0.
  out.append(z)
 return out
def run():
 started=time.time();MET=[];DEP=[];SINGLE=[];fitinfo=[];normal_counts=[]
 for b in ['2.4GHz','5.8GHz']:
  allsets=sets(b)
  for g in ['R','Z','F','F5']:
   for s in [2026,2027,2028]:
    tag=f'{g}_{b}_seed{s}';ck=torch.load(CACHE/(tag+'_best.pt'),map_location='cpu',weights_only=False);m=(ResNet18(1,2) if g=='R' else Fusion(5 if g=='F5' else 10)).cuda().to(memory_format=torch.channels_last);m.load_state_dict(ck['model']);m.eval();norm=ck['normalization'];fitinfo.append({k:ck[k] for k in ['band','group','seed','epoch','val_fa_v3','val_fa_bgA','val_miss','val_loss','thr_fa1','parameters']})
    for name,(x,raw,rs) in allsets.items():
     f=transform(raw,norm);p=predict(m,x,f,g);base=metrics(b,g,s,name,rs,p,ck['thr_fa1']);MET+=base
     writecsv(OUT/(tag+'_'+name+'_predictions.csv'),[dict(r,p_drone=float(p[i]),pred_argmax=int(p[i]>.5),pred_thr=int(p[i]>=ck['thr_fa1'])) for i,r in enumerate(rs)])
     if g=='R' and s==2026:
      z=(np.where(np.isfinite(raw),raw,np.array(norm['median']))-norm['mean'])/norm['std']
      for j,n in enumerate(NAMES):normal_counts.append(dict(band=b,set=name,feature=n,missing=int((~np.isfinite(raw[:,j])).sum()),clipped=int((abs(z[:,j])>10).sum())))
     if g in ['F','F5'] and name in ['test','C1']:
      for method in ['shuffle_all','shuffle_label','zero']:
       rng=np.random.default_rng(20261112)
       for rep in range(1 if method=='zero' else 5):
        ff=f.copy()
        if method=='zero':ff[:]=0
        elif method=='shuffle_all':ff=f[rng.permutation(len(f))]
        else:
         for lab in ['background','drone']:
          ii=np.array([i for i,r in enumerate(rs) if r['label']==lab],dtype=int);ff[ii]=f[rng.permutation(ii)]
        pp=predict(m,x,ff,g);mm=metrics(b,g,s,name,rs,pp,ck['thr_fa1'],method,rep)
        lookup={(r['mode'],r['category'],r['subset']):r['rate'] for r in base}
        mm=[r for r in mm if r['category']!='confusion']
        for r in mm:r['delta']=r['rate']-lookup[r['mode'],r['category'],r['subset']]
        DEP+=mm
     if g in ['F','F5'] and name in ['test','bgA_test','C1']:
      for j in (F5COLS if g=='F5' else range(10)):
       rng=np.random.default_rng(20261116)
       for rep in range(5):
        ff=f.copy();ff[:,j]=f[rng.permutation(len(f)),j];pp=predict(m,x,ff,g)
        lookup={(r['mode'],r['subset']):r['rate'] for r in base if r['category']=='main'}
        for r in metrics(b,g,s,name,rs,pp,ck['thr_fa1']):
         if r['category']=='main' and r['subset'] in ['v3_test_FA','v3_drone_recall','nonG_miss','bgA_test_FA','C1_FA']:
          SINGLE.append(dict(r,feature=NAMES[j],repeat=rep,delta=r['rate']-lookup[r['mode'],r['subset']]))
    del m;torch.cuda.empty_cache();print('EVALUATED',tag,flush=True);writecsv(OUT/'metrics_all.csv',MET);writecsv(OUT/'dependence_all.csv',DEP);writecsv(OUT/'best_models.csv',fitinfo)
  # Audit train/val preprocessing with the same saved parameters.
  norm=json.loads((OUT/(b+'_normalization.json')).read_text(encoding='utf-8'))
  for split in ['train','val']:
   tx,ty,raw,tr,tk=joined(b,split);z=(np.where(np.isfinite(raw),raw,np.array(norm['median']))-norm['mean'])/norm['std']
   for j,n in enumerate(NAMES):normal_counts.append(dict(band=b,set=split,feature=n,missing=int((~np.isfinite(raw[:,j])).sum()),clipped=int((abs(z[:,j])>10).sum())))
   del tx
  # LR10 follows B1's original no-mixture dataset, subgroup sample weighting, val AUROC C selection.
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import roc_auc_score
  xx=[];ff=[];rr=[]
  for split in ['train','val','test']:
   x,f,r=loadset(b+'_'+split)
   if b=='2.4GHz':
    _,bf,br=loadset('bgA');ii=[i for i,z in enumerate(br) if z['split']==split];f=np.concatenate([f,bf[ii]]);r+=[br[i] for i in ii]
   ff.append(f);rr+=r
  f=np.concatenate(ff);y=np.array([r['label']=='drone' for r in rr],dtype=int);tr=np.array([r['split']=='train' for r in rr]);va=np.array([r['split']=='val' for r in rr]);te=np.array([r['split']=='test' for r in rr]);med=np.nanmedian(f[tr],axis=0);med=np.nan_to_num(med);z=np.where(np.isfinite(f),f,med);sc=StandardScaler().fit(z[tr]);z=sc.transform(z);weights=np.zeros(len(y))
  for mask,mass in [(y==1,1),(np.array([r['dataset']=='v3' for r in rr])&(y==0),.5 if b=='2.4GHz' else 1)]+([(np.array([r['dataset']=='dataA_mirror' for r in rr]),.5)] if b=='2.4GHz' else []):weights[tr&mask]=mass/sum(tr&mask)
  weights[tr]/=weights[tr].mean();best=None
  for c in [.01,.1,1,10]:
   lr=LogisticRegression(C=c,max_iter=3000,random_state=2026).fit(z[tr],y[tr],sample_weight=weights[tr]);score=roc_auc_score(y[va],lr.predict_proba(z[va])[:,1])
   if best is None or score>best[0]:best=(score,c,lr)
  pp=best[2].predict_proba(z[te])[:,1];rs=[r for i,r in enumerate(rr) if te[i]];vprob=best[2].predict_proba(z[va])[:,1];bg=np.sort(vprob[y[va]==0]);thr=float(np.nextafter(bg[-1-int(.01*len(bg))],np.inf));MET+=metrics(b,'LR10',2026,'test',rs,pp,thr);dump(OUT/(b+'_LR10_fit.json'),dict(C=best[1],val_auc=best[0],median=med.tolist(),mean=sc.mean_.tolist(),std=sc.scale_.tolist(),coef=best[2].coef_.tolist(),intercept=best[2].intercept_.tolist()))
 writecsv(OUT/'metrics_by_receiver.csv',[r for r in MET if r['category']=='receiver']);writecsv(OUT/'metrics_all.csv',MET);writecsv(OUT/'normalization_counts.csv',normal_counts);writecsv(OUT/'metrics_summary.csv',aggregate(MET,['band','group','set','mode','category','subset','n']));writecsv(OUT/'dependence_summary.csv',aggregate(DEP,['band','group','seed','set','mode','category','subset','dependence'],['rate','delta']));writecsv(OUT/'feature_permutation_single_raw.csv',SINGLE);writecsv(OUT/'feature_permutation_single.csv',aggregate(SINGLE,['band','group','set','mode','subset','feature'],['rate','delta']));dump(OUT/'evaluation_done.json',dict(models=24,seconds=time.time()-started))
if __name__=='__main__':run()
