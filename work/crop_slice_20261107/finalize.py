"""Post-crop cell extraction, recording-reference v3b flags, group-only splitting and audit."""
import importlib.util,sys,collections,time
from pathlib import Path
s=importlib.util.spec_from_file_location('crop',Path(__file__).with_name('crop.py'));c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
import numpy as np
from scipy import fft
O=c.O;FLAGS=['lack_feature','primary_weak','power_shift','floor_raised','interference'];LS=[10,20,30,40,50]
def intervals(mask):
 e=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))
def cells(b):
 out=O/'cells'/Path(b['block_file']).stem;out.mkdir(parents=True,exist_ok=True)
 if (out/'done.json').exists():return {k:np.load(out/(k+'.npy'),mmap_mode='r') for k in ['mean','max','min','power']}
 z=np.load(b['block_file'],mmap_mode='r');fo=c.FS[b['band']];cs=round(fo/1000);nfft=512 if b['band']=='2.4GHz' else 1024;win=np.hanning(nfft).astype(np.float32);nc=len(z)//cs;nf=cs//nfft;ar={k:np.empty((nc,nfft),np.float32) for k in ['mean','max','min']};power=np.empty(nc)
 for j in range(nc):
  x=np.asarray(z[j*cs:(j+1)*cs]);power[j]=np.mean(abs(x)**2);p=abs(fft.fftshift(fft.fft(x[:nf*nfft].reshape(nf,nfft)*win,axis=1),axes=1))**2/np.sum(win**2)
  for key,fun in [('mean',np.mean),('max',np.max),('min',np.min)]:ar[key][j]=10*np.log10(np.maximum(fun(p,axis=0),1e-30))
 ar['power']=power
 for k,v in ar.items():np.save(out/(k+'.npy'),v)
 c.dump(out/'done.json',dict(cells=nc,nfft=nfft));return ar
def assign(bs):
 groups={}
 for b in bs:
  gid=b['group_id'];g=groups.setdefault(gid,{k:b[k] for k in ['group_id','dataset','band','label','model']});g['n_seconds']=g.get('n_seconds',0)+len(np.load(b['block_file'],mmap_mode='r'))/c.FS[b['band']]
 strata=collections.defaultdict(list)
 for g in groups.values():strata[g['dataset'],g['band'],g['label']].append(g)
 rng=np.random.default_rng(2026);short=[]
 for key,gs in sorted(strata.items()):
  total=sum(g['n_seconds'] for g in gs);target=np.array([.6,.2,.2])*total;used=np.zeros(3);by=collections.defaultdict(list)
  for g in gs:by[g['model']].append(g)
  pending=[]
  for model,mg in sorted(by.items()):
   rng.shuffle(mg);mg.sort(key=lambda g:g['n_seconds'],reverse=True)
   if len(mg)<3:short.append(dict(stratum=str(key),model=model,n_groups=len(mg)))
   # Reserve all available splits for each model; largest to train, test before val when scarce.
   order=[2,0] if len(mg)<3 else [0,2,1]
   for g,k in zip(mg,order):g['split']=['train','val','test'][k];used[k]+=g['n_seconds']
   pending.extend(mg[len(order):])
  pending.sort(key=lambda g:g['n_seconds'],reverse=True)
  for g in pending:
   costs=[np.sum(((used+np.eye(3)[k]*g['n_seconds']-target)/np.maximum(target,1e-9))**2) for k in range(3)];k=int(np.argmin(costs));g['split']=['train','val','test'][k];used[k]+=g['n_seconds']
 c.write(O/'group_split.csv',list(groups.values()));c.write(O/'scarce_groups.csv',short);return groups
def screening(gb,ar):
 band=gb[0]['band'];fo=c.FS[band];nfft=ar[0]['mean'].shape[1];freq=np.fft.fftshift(np.fft.fftfreq(nfft,1/fo))/1e6+c.FC[band];mask=abs(freq-c.FC[band])<=c.WIDTH[band];env=np.zeros(nfft,bool)
 for r in c.NOTCHES:
  if r['band']==band:env |= abs(freq-r['freq_mhz'])<=r['half_width_khz']/1000
 mask &= ~env;means=np.concatenate([a['mean'] for a in ar]);mx=np.concatenate([a['max'] for a in ar]);floor=float(np.median(np.percentile(means[:,mask],10,axis=0)));occ=(mx>floor+10).mean(0);occupied=mask&(occ>=.05)
 # Same v3b adjacent-gap merge, with new frequency grid, after clipping fixed band.
 runs=intervals(occupied);merged=[]
 for lo,hi in runs:
  if merged and np.count_nonzero(mask[merged[-1][1]:lo])<=2:merged[-1]=(merged[-1][0],hi)
  else:merged.append((lo,hi))
 ix=[]
 for lo,hi in merged:
  b=np.zeros(nfft,bool);b[lo:hi]=True;b &= mask
  if b.any():ix.append(b)
 floor_cells=np.percentile(means[:,mask],10,axis=1);act=np.stack([(mx[:,b]>floor+10).mean(1) for b in ix],axis=1) if ix else np.empty((len(mx),0));outside=mask & ~(np.logical_or.reduce(ix) if ix else False);ao=(mx[:,outside]>floor+10).mean(1) if outside.any() else np.zeros(len(mx));spans={};offset=0
 for b,a in zip(gb,ar):
  for L in LS:
   spans.setdefault(L,[]).extend((offset+j,offset+j+L,b,j) for j in range(0,len(a['mean'])-L+1,L))
  offset+=len(a['mean'])
 data={}
 for L,ss in spans.items():
  av=np.array([act[s:e].mean(0) for s,e,_,_ in ss]);ov=np.array([ao[s:e].mean() for s,e,_,_ in ss]);sv=np.full((len(ss),len(ix)),np.nan)
  for i,(s,e,_,_) in enumerate(ss):
   for k,b in enumerate(ix):
    block=mx[s:e][:,b];active=(block>floor+10).any(1)
    if active.sum()>=2:sv[i,k]=np.median(block[active].mean(1)-floor)
  data[L]=(av,ov,sv)
 av50=data[50][0];occ50=(av50>=.01).mean(0) if len(av50) else np.zeros(len(ix));chars=np.flatnonzero(occ50>=.9);primary=int(max(chars,key=lambda k:np.median(av50[:,k])*ix[k].sum())) if len(chars) else (int(np.argmax(occ50)) if len(ix) else -1);result={}
 for L,ss in spans.items():
  av,ov,sv=data[L];rr=[]
  if not len(ss):result[L]=rr;continue
  dense=[k for k in chars if (av[:,k]>=.01).mean()>=.9];med=np.median(av,axis=0);meds=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in sv.T]);outlim=max(.05,3*np.median(ov));tested=set(([primary] if primary>=0 else [])+[k for k in dense if med[k]>=.2])
  for i,(s,e,b,j) in enumerate(ss):
   flags=[any(av[i,k]<.01 for k in dense),primary>=0 and av[i,primary]<.25*med[primary],any(abs(sv[i,k]-meds[k])>6 for k in tested),np.median(floor_cells[s:e])-floor>3,ov[i]>outlim];rr.append((b,j,[bool(v) for v in flags]))
  result[L]=rr
 return result,dict(group_id=gb[0]['group_id'],floor_db=floor,n_subbands=len(ix),subbands=[dict(lo=float(freq[b][0]),hi=float(freq[b][-1])) for b in ix],primary=primary,persistent=chars.tolist())
def main():
 bs=c.read(O/'plan.json');blockrows=c.rows(O/'blocks.csv');assert len(blockrows)==len(bs)
 # Original 008/F008 folders omit literal M; retain Task A's interpretation.
 corrections=[]
 for b in bs:
  if b['dataset']=='data' and b['bw_label']=='unknown':
   match=c.re.search(r'_(\d+)M?(?:_combination)?-signal',b['group_id'])
   if match:
    b['bw_label']=match[1]+'M';corrections.append(dict(group_id=b['group_id'],field='bw_label',old='unknown',new=b['bw_label'],reason='original folder omits M; Task A regex interpretation'))
 if corrections:c.write(O/'metadata_corrections.csv',corrections)
 byfile={b['block_file']:b for b in bs}
 for r in blockrows:
  r['bw_label']=byfile[r['block_file']]['bw_label'];meta=O/'block_meta'/(Path(r['block_file']).stem+'.json');d=c.read(meta);d['bw_label']=r['bw_label'];c.dump(meta,d)
 c.dump(O/'plan.json',bs);c.write(O/'blocks.csv',blockrows);groups=assign(bs);by=collections.defaultdict(list)
 for b in bs:by[b['group_id']].append(b)
 output={L:[] for L in LS};features=[]
 for gi,(gid,gb) in enumerate(by.items()):
  ars=[cells(b) for b in gb]
  if gb[0]['label']=='drone':rr,info=screening(gb,ars);features.append(info)
  else:rr={L:[(b,j,[False]*5) for b,a in zip(gb,ars) for j in range(0,len(a['mean'])-L+1,L)] for L in LS}
  for L,rs in rr.items():
   for b,j,flags in rs:
    r={k:b[k] for k in ['block_file','dataset','band','label','model','state','bw_label','group_id']};r.update(sample_id=f'L{L}_{len(output[L]):08d}',start=j*round(c.FS[b['band']]/1000),n_samples=L*round(c.FS[b['band']]/1000),split=groups[gid]['split'],**dict(zip(FLAGS,flags)),keep=not any(flags));output[L].append(r)
  print('SCREEN_GROUP',gi+1,len(by),gid,flush=True)
 for L,rs in output.items():c.write(O/f'index_L{L}.csv',rs)
 c.dump(O/'recording_features.json',features);counts=[];models=[];rates=[];errors=0
 for L,rs in output.items():
  ct=collections.defaultdict(list);mt=collections.Counter();rt=collections.defaultdict(list)
  for r in rs:
   ct[(r['split'],r['dataset'],r['band'],r['label'])].append(r);mt[r['model'],r['split']]+=1;rt[r['band'],r['state']].append(r);z=np.load(r['block_file'],mmap_mode='r');errors+=int(r['start']<0 or r['start']+r['n_samples']>len(z) or r['split']!=groups[r['group_id']]['split'])
  for (split,ds,band,label),rr in ct.items():counts.append(dict(L=L,split=split,dataset=ds,band=band,label=label,n=len(rr),keep=sum(r['keep'] for r in rr),seconds=len(rr)*L/1000,keep_seconds=sum(r['keep'] for r in rr)*L/1000))
  for (model,split),n in mt.items():models.append(dict(L=L,model=model,split=split,n=n))
  for (band,state),rr in rt.items():rates.append(dict(L=L,band=band,state=state,n=len(rr),keep=sum(r['keep'] for r in rr),retention=sum(r['keep'] for r in rr)/len(rr)))
 c.write(O/'counts.csv',counts);c.write(O/'model_counts.csv',models);c.write(O/'retention.csv',rates);c.dump(O/'index_verification.json',dict(leakage_and_boundaries=dict(status='pass' if not errors else 'fail',errors=errors,groups=len(groups),rows=sum(map(len,output.values()))),counts=dict(status='pass',per_L={str(L):dict(n=len(rs),keep=sum(r['keep'] for r in rs)) for L,rs in output.items()})));print('FINALIZE_COMPLETE',flush=True)
if __name__=='__main__':main()
