from evaluate import *
def oldrefs():
 rr=rows(ROOT/'work/side_L_wifimix_20261110/metrics_all.csv');old=[]
 for r in rr:
  if r['round']=='Lb':old.append(dict(band='2.4GHz',group='OLD_Lb',seed=r['seed'],set='old',mode=r['mode'],category=r['category'],subset=r['group'],n=r['n'],k=r['k'],rate=r['rate'],cp95_upper=r['cp95_upper'],reference='旧 checkpoint，不是本次训练'))
 for s in [2026,2027,2028]:
  rs=rows(ROOT/f'work/train_stage1_L20_20261108/test_predictions_5.8GHz_seed{s}.csv');p=np.array([float(r['p_drone']) for r in rs]);thr=next(float(r['threshold']) for r in rows(ROOT/'work/train_stage1_L20_20261108/main_metrics.csv') if r['band']=='5.8GHz' and int(r['seed'])==s and r['mode']=='thr_fa1');z=metrics('5.8GHz','OLD_H',s,'test',rs,p,thr)
  for r in z:r['reference']='旧 checkpoint，不是本次训练'
  old+=z
 for folder,band,group in [('side_L_wifimix_20261110','2.4GHz','OLD_Lb'),('train_stage1_L20_20261108','5.8GHz','OLD_H')]:
  for r in rows(ROOT/'work'/folder/'mix_results.csv'):
   if (folder.startswith('side_') and r['round']!='Lb') or (not folder.startswith('side_') and r['band']!=band):continue
   n=int(r['n']);rate=float(r.get('rate',r.get('detection_rate')));k=int(round(rate*n));fa=r['group']=='v3_background';old.append(dict(band=band,group=group,seed=r['seed'],set='mix_test',mode=r['mode'],category='mix_FA' if fa else 'mix_recall',subset=r['group']+' | '+r['x_db'],n=n,k=k,rate=rate,cp95_upper=(float(beta.ppf(.95,k+1,n-k)) if k<n else 1.) if fa else '',reference='? checkpoint???????'))
 for r in rows(ROOT/'work/train_stage1_L20_20261108/rejected_test_recall.csv'):
  if r['band']=='5.8GHz':old.append(dict(band='5.8GHz',group='OLD_H',seed=r['seed'],set='rejected_test',mode=r['mode'],category='rejected',subset='all_rejected_recall',n=r['n'],k=round(float(r['recall'])*int(r['n'])),rate=r['recall'],cp95_upper='',reference='? checkpoint???????'))
 for r in rows(ROOT/'work/side_5p8_controls_20261109/inference.csv'):
  if r['band']=='5.8GHz' and r['model']=='H' and r['input'] in ['a','b']:
   n=int(r['n']);k=int(r['k']);old.append(dict(band='5.8GHz',group='OLD_H',seed=r['seed'],set='control',mode=r['decision'],category='main',subset=('white_noise' if r['input']=='a' else 'notched_noise')+'_FA',n=n,k=k,rate=r['rate'],cp95_upper=float(beta.ppf(.95,k+1,n-k)) if k<n else 1.,reference='? checkpoint?????????N=200???N=80'))
 for r in old:
  if r['band']=='2.4GHz' and r['category']=='G_recall':r['subset']='data | '+r['subset']
  if r['category'] in ['bgA_test_by_source','bgA_val_by_source']:
   r['set']=r['category'].replace('_by_source','');r['category']='bgA_by_source'
  if r['category']=='quiet013_by_group_split':r['category']='quiet013_by_group'
  if r['category'].startswith('mix_data_model_'):
   r['subset']='data_drone | '+r['subset']+' | '+r['category'].replace('mix_data_model_','');r['category']='mix_model';r['set']='mix_test'
 for s in [2026,2027,2028]:
  for mode in ['argmax','thr_fa1']:
   z=[r for r in old if r['band']=='2.4GHz' and int(r['seed'])==s and r['mode']==mode and r['category']=='C1'];n=sum(int(r['n']) for r in z);k=sum(int(r['k']) for r in z)
   if n:old.append(dict(band='2.4GHz',group='OLD_Lb',seed=s,set='C1',mode=mode,category='main',subset='C1_FA',n=n,k=k,rate=k/n,cp95_upper=float(beta.ppf(.95,k+1,n-k)) if k<n else 1.,reference='旧 checkpoint，不是本次训练'))
 for r in old:r['reference']='旧 checkpoint，不是本次训练'
 writecsv(OUT/'old_reference_metrics.csv',old)
def plot():
 rr=rows(OUT/'metrics_summary.csv');rr=[r for r in rr if r['group'] in ['R','Z','F','F5'] and r['mode']=='argmax'];colors=['#4c78a8','#f2a541','#59a14f','#b279a2']
 def get(b,g,category,subset):
  rs=[r for r in rr if r['band']==b and r['group']==g and r['category']==category and r['subset']==subset]
  return (float(rs[0]['rate_mean']),float(rs[0]['rate_std'])) if rs else (np.nan,0)
 fig,axs=plt.subplots(2,2,figsize=(19,11))
 for row,b in enumerate(['2.4GHz','5.8GHz']):
  left=[('v3 test背景','main','v3_test_FA')]+([('bgA test','main','bgA_test_FA'),('C1合并','main','C1_FA'),('013安静','main','quiet013_FA')] if row==0 else [])+[('白噪声','main','white_noise_FA'),('陷波白噪声','main','notched_noise_FA')]
  right=[('data非G','main','data_nonG_recall'),('v3无人机','main','v3_drone_recall'),('mavic2','v3_model_recall','mavic2'),('mini3pro','v3_model_recall','mini3pro')]
  right +=[(r['subset'],'G_recall',r['subset']) for r in rr if r['band']==b and r['group']=='R' and r['category']=='G_recall']
  for ax,items,title in zip(axs[row],[left,right],['虚警','召回']):
   for j,g in enumerate(['R','Z','F','F5']):
    v=[get(b,g,cat,sub) for lab,cat,sub in items];ax.bar(np.arange(len(items))+(j-1.5)*.2,[z[0]*100 for z in v],.2,yerr=[z[1]*100 for z in v],label=g,color=colors[j],capsize=3)
   ax.set_xticks(range(len(items)),[r[0] for r in items],rotation=25,ha='right');ax.set(title=b+' '+title,ylabel=title+'率 (%)',ylim=(0,105));ax.legend()
 fig.suptitle('时频图 + IQ 特征：R / Z / F / F5，三种子均值±标准差（argmax）');fig.tight_layout();fig.savefig(OUT/'图20_时频图加IQ特征_一阶段对比.png',dpi=160);plt.close(fig)
 dep=rows(OUT/'dependence_all.csv');fig,axs=plt.subplots(1,2,figsize=(13,5))
 for ax,b in zip(axs,['2.4GHz','5.8GHz']):
  for j,method in enumerate(['shuffle_all','shuffle_label','zero']):
   vals=[];errs=[]
   for subset in ['v3_drone_recall','nonG_miss']:
    z=[float(r['delta'])*100 for r in dep if r['band']==b and r['set']=='test' and r['mode']=='argmax' and r['subset']==subset and r['dependence']==method];vals.append(np.mean(z));errs.append(np.std(z,ddof=1))
   ax.bar(np.arange(2)+(j-1)*.24,vals,.24,yerr=errs,label={'shuffle_all':'全体打乱','shuffle_label':'标签内打乱','zero':'置零'}[method],capsize=3)
  ax.axhline(0,color='black',lw=.7);ax.set_xticks([0,1],['v3无人机召回','非G漏检']);ax.set(title=b,ylabel='相对F变化（百分点）');ax.legend()
 fig.tight_layout();fig.savefig(OUT/'fig_feature_dependence.png',dpi=160);plt.close(fig)
 for b in ['2.4GHz','5.8GHz']:
  fig,axs=plt.subplots(1,3,figsize=(17,5))
  for ax,kind in zip(axs,['data_drone','v3_drone','v3_background']):
   for g in ['R','Z','F','F5']:
    cat='mix_FA' if kind=='v3_background' else 'mix_recall';xs=[-10,-5,0,5,10,15,20];v=[get(b,g,cat,kind+' | '+str(db)) for db in xs];ax.errorbar(xs,[z[0] for z in v],yerr=[z[1] for z in v],label=g,marker='o')
   ax.axvspan(-5,15,color='gray',alpha=.1);ax.set(title=kind,xlabel='叠加功率比 X (dB)',ylabel='检出率 / 虚警率',ylim=(-.03,1.05));ax.legend()
  fig.tight_layout();fig.savefig(OUT/f'fig3_mix_test_{b}.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5))
 for g in ['R','Z','F','F5']:
  xs=[-10,-5,0,5,10,15,20];v=[get('2.4GHz',g,'C1',str(db)) for db in xs];ax.errorbar(xs,[z[0] for z in v],yerr=[z[1] for z in v],label=g,marker='o')
 ax.set(xlabel='叠加功率比 X (dB)',ylabel='C1 虚警率',ylim=(-.03,1.05));ax.legend();fig.tight_layout();fig.savefig(OUT/'fig_C1_2.4GHz.png',dpi=150);plt.close(fig)
 for b in ['2.4GHz','5.8GHz']:
  fig,axs=plt.subplots(1,3,figsize=(16,4))
  for g in ['R','Z','F','F5']:
   for s in [2026,2027,2028]:
    rs=rows(OUT/f'{g}_{b}_seed{s}_epochs.csv')
    for ax,k in zip(axs,['val_fa_v3','val_miss','val_loss']):ax.plot([int(r['epoch']) for r in rs],[float(r[k]) for r in rs],label=g+str(s));ax.set(xlabel='Epoch',ylabel=k)
  axs[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(OUT/f'fig5_curves_{b}.png',dpi=150);plt.close(fig)
def extras():
 from common import image
 wifi=[]
 for split in ['train','val','test']:
  x,f,rr=loadset('2.4GHz_'+split);freq=2405+(np.arange(256)+.5)*50/256
  for gid in sorted({r['group_id'] for r in rr if r['label']=='background'}):
   ids=[i for i,r in enumerate(rr) if r['group_id']==gid and r['label']=='background'];profile=np.median(x[ids].astype(float),axis=2);ref=profile[:,(freq>=2451)&(freq<=2454)].mean(1);wifi.append(dict(split=split,group_id=gid,n=len(ids),ch1_minus_ref=float(np.median(profile[:,(freq>=2408)&(freq<=2416)].mean(1)-ref)),ch6_minus_ref=float(np.median(profile[:,(freq>=2433)&(freq<=2441)].mean(1)-ref))))
 writecsv(OUT/'wifi_coarse.csv',wifi)
 for b in ['2.4GHz','5.8GHz']:
  x,f,rs=loadset(b+'_mix_test');orig,_,rr=loadset(b+'_test');lookup={r['sample_id']:i for i,r in enumerate(rr)};sid=next(r['sample_id'] for r in rs if r['group']=='data_drone');chosen=[next(i for i,r in enumerate(rs) if r['sample_id']==sid and int(r['x_db'])==db) for db in [-5,5,15]];bgid=rs[chosen[0]]['background_sample_id'];fig,axs=plt.subplots(1,4,figsize=(16,4));image(axs[0],orig[lookup[bgid]],b,'原始背景')
  for ax,i in zip(axs[1:],chosen):image(ax,x[i],b,rs[i]['x_db']+' dB')
  fig.tight_layout();fig.savefig(OUT/f'fig3b_mix_examples_{b}.png',dpi=140);plt.close(fig)
def auxiliary():
 from preflight import INDEX
 power=[];start=time.time()
 for r in sorted([r for r in INDEX if r['split']=='test'],key=lambda r:(r['block_file'],int(r['start']))):
  z=iq(r);n=int(r['n_samples'])//20;db=10*np.log10(np.maximum(np.mean(abs(z.reshape(20,n))**2,axis=1),1e-30));q10,q90=np.quantile(db,[.1,.9]);power.append(dict(r,power_p10_db=float(q10),power_p90_db=float(q90),power_proxy_db=float(q90-q10)))
 writecsv(OUT/'power_proxy.csv',power);fig,axs=plt.subplots(1,2,figsize=(12,4))
 for ax,b in zip(axs,['2.4GHz','5.8GHz']):
  for ds in ['data','v3']:
   vals=[r['power_proxy_db'] for r in power if r['band']==b and r['dataset']==ds and r['label']=='drone' and r['keep']=='True'];ax.hist(vals,bins=25,alpha=.5,label=ds)
  ax.set(title=b,xlabel='1ms??P90?P10 (dB)',ylabel='??');ax.legend()
 fig.tight_layout();fig.savefig(OUT/'fig_power_proxy.png',dpi=150);plt.close(fig)
 new=rows(OUT/'metrics_summary.csv');old=rows(OUT/'old_reference_metrics.csv');out=[]
 for r in new:
  if r['group'] not in ['R','Z','F','F5']:continue
  ref=[z for z in old if z['band']==r['band'] and z['mode']==r['mode'] and z['category']==r['category'] and z['subset']==r['subset'] and (r['category']!='bgA_by_source' or z['set']==r['set'])]
  v=[float(z['rate']) for z in ref];out.append(dict(r,old_checkpoint_mean=float(np.mean(v)) if v else '',old_checkpoint_std=float(np.std(v,ddof=1)) if len(v)>1 else '',reference='? checkpoint???????'))
 for r in out:r['reference']='旧 checkpoint，不是本次训练'
 writecsv(OUT/'comparison.csv',out);ref=[]
 for b in ['2.4GHz','5.8GHz']:
  for seed in [2026,2027,2028]:
   oldpath=ROOT/(f'work/side_L_wifimix_20261110/train_log_Lb_seed{seed}.csv' if b=='2.4GHz' else f'work/train_stage1_L20_20261108/train_log_{b}_seed{seed}.csv');a=rows(oldpath);z=rows(OUT/f'R_{b}_seed{seed}_epochs.csv')
   for ar,zr in zip(a,z):ref.append(dict(band=b,seed=seed,epoch=zr['epoch'],old_train_loss=ar['train_loss'],new_train_loss=zr['train_loss'],old_val_loss=ar['val_loss'],new_val_loss=zr['val_loss'],old_seconds=ar['seconds'],new_seconds=zr['seconds']))
 writecsv(OUT/'training_reference.csv',ref)
def report():
 from finish_b2b import report as current_report
 current_report()
if __name__=='__main__':
 from finish_b2b import newplots,supplemental
 oldrefs();plot();extras();auxiliary();newplots();supplemental();report()
