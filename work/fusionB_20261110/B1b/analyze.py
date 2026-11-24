import sys,os,time,json
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O))
from screen import *
from generate import snapshot
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False

def auc(a,b):
 a=a[np.isfinite(a)];b=b[np.isfinite(b)]
 if not len(a) or not len(b):return np.nan,np.nan
 v=roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))],np.r_[a,b]);return max(v,1-v),v

def comparison(a,b,groups=None):
 out=[];gg=[] if groups is None else [g for g in sorted(set(groups)) if sum(groups==g)>=3]
 for j,name in enumerate(NAMES):
  v,raw=auc(a[:,j],b[:,j]);pa=np.nanpercentile(a[:,j],[5,50,95]);pb=np.nanpercentile(b[:,j],[5,50,95]) if len(b) else [np.nan]*3
  block=[auc(a[:,j],b[groups==g,j])[0] for g in gg];lo=min(block) if block else np.nan;hi=max(block) if block else np.nan
  mark='接收机指纹' if v>=.90 and lo>=.80 else ('可疑' if v>=.75 else '未见接收机差异')
  out.append(dict(feature=name,auc=v,raw_auc=raw,direction='B高' if raw>.5 else ('B低' if raw<.5 else '相同'),n0=len(a),n1=len(b),A_p5=pa[0],A_median=pa[1],A_p95=pa[2],B_p5=pb[0],B_median=pb[1],B_p95=pb[2],block_min=lo,block_max=hi,mark=mark))
 return pd.DataFrame(out)

def main():
 t=time.time();meta={n:pd.read_csv(O/f'{n}_index.csv',dtype={'nn':str}) for n in ['bgA','bgB','A013','B2440']};meta['bgA']=pd.read_csv(O/'bgA_analysis_index.csv');x={n:np.load(O/f'{n}_features.npy') for n in meta};q={n:meta[n].quiet.to_numpy(bool) for n in meta}
 a=x['bgA'][q['bgA']];b=x['bgB'][q['bgB']];groups=meta['bgB'].nn.to_numpy()[q['bgB']]
 main=comparison(a,b,groups);app1=comparison(x['bgA'],x['bgB']);app2=comparison(x['A013'],x['B2440'][q['B2440']])

 app3meta={n:pd.read_csv(O/f'appendix3_{n}_index.csv') for n in ['bgA','bgB']};app3x={n:np.load(O/f'original_{n}_features.npy') for n in ['bgA','bgB']};app3=comparison(app3x['bgA'][app3meta['bgA'].quiet],app3x['bgB'][app3meta['bgB'].quiet]);app3.to_csv(O/'appendix3_no_notch.csv',index=False,encoding='utf-8')
 for n,df in [('main_table',main),('appendix1_all',app1),('appendix2_inband',app2)]:df.to_csv(O/f'{n}.csv',index=False,encoding='utf-8')
 per=[]
 for g in sorted(set(groups)):
  if sum(groups==g)<3:continue
  tab=comparison(a,b[groups==g]);tab['nn']=g;per.append(tab)
 pd.concat(per).to_csv(O/'auc_per_block.csv',index=False,encoding='utf-8')
 # Five folds assigned once; every A and B row receives exactly one OOF prediction.
 allg=meta['bgB'].nn.unique();folds=list(KFold(5,shuffle=True,random_state=20261110).split(a));preda=np.full(len(a),np.nan);predb=np.full(len(b),np.nan);fitlog=[]
 for k,g in enumerate(allg):
  tr,te=folds[k];bt=groups==g;trainx=np.r_[a[tr,:13],b[~bt,:13]];trainy=np.r_[np.zeros(len(tr)),np.ones(sum(~bt))]
  model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=20261110));model.fit(trainx,trainy);preda[te]=model.predict_proba(a[te,:13])[:,1]
  if bt.any():predb[bt]=model.predict_proba(b[bt,:13])[:,1]
  fitlog.append(dict(fold=k,held_nn=g,A_test=len(te),B_test=int(bt.sum()),iterations=int(model[-1].n_iter_[0])))
 assert np.isfinite(preda).all() and np.isfinite(predb).all();oof=float(roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))],np.r_[preda,predb]));dump('lr_oof.json',dict(auc=oof,symmetric_auc=max(oof,1-oof),folds=fitlog,seed=20261110,features=NAMES[:13],C=1,class_weight='balanced'))
 pd.DataFrame(dict(side=['A']*len(a)+['B']*len(b),prediction=np.r_[preda,predb],sample_id=list(meta['bgA'][q['bgA']].sample_id)+list(meta['bgB'][q['bgB']].sample_id))).to_csv(O/'lr_oof_predictions.csv',index=False,encoding='utf-8')
 rng=np.random.default_rng(20261110);ar=meta['bgA'][q['bgA']];br=meta['bgB'][q['bgB']];ar=ar.iloc[rng.choice(len(ar),min(200,len(ar)),replace=False)]
 def meanspec(rr):
  return np.mean([np.array([spectrum(getz(r))[:,ix].mean() for ix in BINS]) for r in rr.to_dict('records')],axis=0)
 # Compute each FFT once per selected segment.
 def meanspec(rr):
  sums=[]
  for r in rr.to_dict('records'):
   ps=spectrum(getz(r)).mean(0);sums.append([ps[ix].mean() for ix in BINS])
  v=np.mean(sums,axis=0);return 10*np.log10(v/np.median(v))
 sa=meanspec(ar);sb=meanspec(br);freq=np.arange(50)+2405.5;pd.DataFrame(dict(freq_mhz=freq,A_db=sa,B_db=sb,difference_B_minus_A_db=sb-sa)).to_csv(O/'mean_spectrum.csv',index=False,encoding='utf-8')
 fig=plt.figure(figsize=(19,12));gs=fig.add_gridspec(4,3,width_ratios=[1.25,1.05,1],wspace=.32,hspace=.8)
 ax=fig.add_subplot(gs[:2,0]);ax.plot(freq,sa,label=f'bgA 安静 (n={len(ar)})');ax.plot(freq,sb,label=f'bgB 安静 (n={len(br)})');ax.legend();ax.set(title='1 MHz 平均功率谱',xlabel='MHz',ylabel='相对通带中位数 / dB');ax.grid(alpha=.2)
 ax=fig.add_subplot(gs[2:,0]);ax.plot(freq,sb-sa,color='purple');ax.axhline(0,color='gray',lw=.7);ax.set(title='bgB − bgA',xlabel='MHz',ylabel='dB');ax.grid(alpha=.2)
 examples=[('bgA',ar.iloc[0]),('bgA',ar.iloc[-1]),('bgB',br.iloc[0]),('bgB',br.iloc[-1])]
 for i,(label,r) in enumerate(examples):
  c=cells(getz(r));db=10*np.log10(c/np.median(c));ax=fig.add_subplot(gs[i,1]);im=ax.imshow(db.T,origin='lower',aspect='auto',extent=[0,20,2405,2455],vmin=-3,vmax=25,cmap='viridis');ax.set(title=f'{label}: {r.sample_id}',xlabel='ms',ylabel='MHz');fig.colorbar(im,ax=ax,label='dB',pad=.02)
 for i,name in enumerate(['shape_factor','spec_flatness','sigma_f','c20n_abs']):
  j=NAMES.index(name);ax=fig.add_subplot(gs[i,2]);ax.boxplot([a[:,j],b[:,j]],tick_labels=['bgA','bgB'],showfliers=False);ax.set_title(name);ax.grid(alpha=.2)
 fig.suptitle('图17 接收机 A 与 B 镜像背景对比（同一安静筛选规则）',fontsize=16,y=.99);fig.savefig(O/'图17_接收机A与B镜像背景对比.png',dpi=160,bbox_inches='tight');plt.close(fig)
 dump('analysis_runtime.json',dict(seconds=time.time()-t));print('ANALYSIS_DONE',len(a),len(b),oof,flush=True)
 after=snapshot();dump('hash_after.json',after);before=json.loads((O/'hash_before.json').read_text(encoding='utf-8'));changed=[k for k in set(before)|set(after) if before.get(k)!=after.get(k)];dump('hash_verification.json',dict(n_before=len(before),n_after=len(after),identical=not changed,changed=changed));assert not changed,changed
 print('HASH_IDENTICAL',len(after),flush=True)
if __name__=='__main__':main()