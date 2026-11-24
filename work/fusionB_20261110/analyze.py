import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,csv,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from extract import ROOT,OUT,CACHE,NAMES,rows,writecsv,dump
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
CS=[.01,.1,1,10]
def save(name,rr):writecsv(OUT/name,rr)
def load():
    allrows=[];xx=[]
    for p in sorted(CACHE.glob('*_features_meta.csv')):
        rr=rows(p);name=p.name.replace('_features_meta.csv','');x=np.load(CACHE/(name+'_features.npy'))
        for r in rr:r['meta_name']=name
        allrows.extend(rr);xx.extend(x)
    return pd.DataFrame(allrows),np.array(xx)
def fill(x,train):
    with np.errstate(all='ignore'):med=np.nanmedian(x[train],axis=0)
    med=np.where(np.isfinite(med),med,0);return np.where(np.isfinite(x),x,med),med
def auc(y,v):
    if len(np.unique(y))<2:return float('nan'),'不可估'
    a=roc_auc_score(y,v);return max(a,1-a),('组1大' if a>=.5 else '组0大')
def univ(x,mask,y,title,extra=None):
    ii=np.flatnonzero(np.asarray(mask));y=np.asarray(y)[ii];out=[]
    for j,n in enumerate(NAMES):
        a,d=auc(y,x[ii,j]);out.append(dict(comparison=title,feature=n,auc=a,direction=d,n0=int(sum(y==0)),n1=int(sum(y==1)),**(extra or {})))
    return out
def trainlr(x,y,tr,va,cols,weights=None):
    # Fit train-only imputer and scaler, including all-NaN->0 fallback.
    z,med=fill(x[:,cols],tr);sc=StandardScaler().fit(z[tr]);z=sc.transform(z);best=None
    for C in CS:
        mod=LogisticRegression(C=C,max_iter=3000,random_state=2026).fit(z[tr],y[tr],sample_weight=None if weights is None else weights[tr])
        p=mod.predict_proba(z[va])[:,1];score=roc_auc_score(y[va],p) if len(np.unique(y[va]))==2 else -log_loss(y[va],p,labels=[0,1])
        if best is None or score>best[0]:best=(score,C,mod)
    return best[2].predict_proba(z)[:,1],best[1],int(sum(~np.isfinite(np.nanmedian(x[tr][:,cols],axis=0))))
def isG(df):
    return ((df.dataset=='data')&(df.band=='2.4GHz')&df.model.isin(['evo lite 6k','evo ii pro v3']))|((df.dataset=='v3')&(df.band=='5.8GHz')&df.model.isin(['evo2pro_v3','fpv']))
def metrics(df,p,band,model):
    df=df.reset_index(drop=True);p=np.asarray(p);pred=p>=.5;dr=(df.label=='drone').values;g=isG(df).values
    out=[]
    def add(mask,metric,group):
        mask=np.asarray(mask);N=int(mask.sum());k=int(np.sum((pred if metric=='FA' else (~pred if metric=='miss' else pred))[mask]));upper=float(beta.ppf(.95,k+1,N-k)) if metric=='FA' and k<N else (1. if metric=='FA' and N else np.nan)
        out.append(dict(band=band,model=model,metric=metric,group=group,k=k,n=N,rate=k/N if N else np.nan,cp95_upper=upper))
    bg=(df.label=='background').values;vb=(df.dataset=='v3').values&bg;ab=(df.dataset=='dataA_mirror').values&bg
    add(vb,'FA','v3_all');
    for group in sorted(df.loc[vb,'group_id'].unique()):add(vb&(df.group_id==group).values,'FA',group)
    if band=='2.4GHz':add(ab,'FA','bgA_all')
    add(dr&~g,'miss','nonG');add(dr,'miss','includingG')
    for dataset in ['data','v3']:add(dr&(df.dataset==dataset).values,'recall',dataset)
    for dataset in ['data','v3']:
        for m in sorted(df.loc[dr&(df.dataset==dataset),'model'].unique()):add(dr&(df.dataset==dataset).values&(df.model==m).values,'recall',dataset+':'+m)
    for dataset,m in [('data','evo lite 6k'),('data','evo ii pro v3'),('v3','evo2pro_v3'),('v3','fpv')]:
        mask=g&(df.dataset==dataset).values&(df.model==m).values
        if mask.any():add(mask,'recall','G:'+dataset+':'+m)
    return out
def source_b(df,x):
    records=[];multi=[];bwrecords=[]
    rawroot=Path('G:/drone/raw');bwmap={}
    gids=df.loc[(df.dataset=='v3')&(df.model=='mini5pro'),'group_id'].unique()
    # Inventory JSON filenames once, then restrict reads to requested sessions.
    for gid in gids:
        sid=gid.split(':',1)[-1];values=[];paths=[]
        candidates=[]
        for campaign in rawroot.iterdir():
            if campaign.is_dir():
                session=campaign/sid
                if session.is_dir():candidates.extend(session.glob('*.json'))
        direct=rawroot/sid
        if direct.is_dir():candidates.extend(direct.glob('*.json'))
        for path in sorted(set(candidates)):
            obj=json.loads(path.read_text(encoding='utf-8-sig'));bw=obj.get('declared',{}).get('bw')
            if isinstance(bw,list):
                values.extend(str(v) for v in bw if v is not None)
            elif bw is not None:values.append(str(bw))
            paths.append(str(path))
        bwmap[gid]=values[0] if len(set(values))==1 else 'unknown'
        bwrecords.append(dict(group_id=gid,bw=bwmap[gid],values=';'.join(values),json_files=';'.join(paths)))
    save('v3_declared_bw.csv',bwrecords)
    def norm(v):
        v=str(v).lower().replace('mhz','').replace('m','').strip()
        try:
            a=float(v);return str(int(a/1e6 if a>1e5 else a))+'M'
        except ValueError:return 'unknown'
    for band,da,v3 in [('2.4GHz','mini5 pro','mini5pro'),('5.8GHz','mini5 pro','mini5pro'),('2.4GHz','evo ii pro v3','evo2pro_v3')]:
        mask=(df.keep=='True')&~df.meta_name.str.contains('rejected')&(df.band==band)&(((df.dataset=='data')&(df.model==da))|((df.dataset=='v3')&(df.model==v3)))
        subsets=[('pooled',mask)]
        if v3=='mini5pro':
            bw=np.array([norm(bwmap.get(r.group_id,'unknown') if r.dataset=='v3' else r.bw_label) for r in df.itertuples()])
            common=set(bw[mask&(df.dataset=='data')])&set(bw[mask&(df.dataset=='v3')]);common.discard('unknown')
            for b in sorted(common):subsets.append((b,mask&(bw==b)))
            remaining=mask&~np.isin(bw,list(common))
            if remaining.any():subsets.append(('unmatched',remaining))
        for b,m in subsets:
            title=f'{band}:{v3}:{b}'
            ux,_=fill(x,((df.band==band)&(df.split=='train')&(df.keep=='True')&~df.meta_name.str.contains('rejected')).values)
            records+=univ(ux,m,(df.dataset=='v3').astype(int),title)
            ii=np.flatnonzero(m);y=(df.iloc[ii].dataset=='v3').astype(int).values;groups=df.iloc[ii].group_id.values
            for ncols in [13,19]:
                preds=np.full(len(ii),np.nan);selected=[];skip=0
                if len(np.unique(groups))>=2:
                    for tr,te in LeaveOneGroupOut().split(x[ii],y,groups):
                        if len(np.unique(y[tr]))<2:skip+=len(te);continue
                        # Existing val, restricted to remaining training groups. If absent/one class,
                        # use grouped inner cross validation to tune C, never held-out data.
                        va=tr[(df.iloc[ii[tr]].split=='val').values];fit=tr[(df.iloc[ii[tr]].split!='val').values]
                        if len(va) and len(np.unique(y[va]))==2 and len(np.unique(y[fit]))==2:
                            trmask=np.zeros(len(ii),bool);trmask[fit]=1;vamask=np.zeros(len(ii),bool);vamask[va]=1
                            pp,C,_=trainlr(x[ii],y,trmask,vamask,list(range(ncols)))
                            # Refit on all outer training groups after selecting C.
                        else:
                            scores=[]
                            for C0 in CS:
                                ip=np.full(len(tr),np.nan)
                                if len(np.unique(groups[tr]))>1:
                                    for a,b0 in LeaveOneGroupOut().split(x[ii[tr]],y[tr],groups[tr]):
                                        if len(np.unique(y[tr[a]]))<2:continue
                                        z,_=fill(x[ii][:,:ncols],tr[a]);sc=StandardScaler().fit(z[tr[a]]);lr=LogisticRegression(C=C0,max_iter=3000).fit(sc.transform(z[tr[a]]),y[tr[a]]);ip[b0]=lr.predict_proba(sc.transform(z[tr[b0]]))[:,1]
                                ok=np.isfinite(ip);scores.append(roc_auc_score(y[tr][ok],ip[ok]) if ok.any() and len(np.unique(y[tr][ok]))==2 else -np.inf)
                            C=CS[int(np.argmax(scores))] if np.isfinite(scores).any() else 1.
                        z,_=fill(x[ii][:,:ncols],tr);sc=StandardScaler().fit(z[tr]);lr=LogisticRegression(C=C,max_iter=3000).fit(sc.transform(z[tr]),y[tr]);preds[te]=lr.predict_proba(sc.transform(z[te]))[:,1];selected.append(C)
                good=np.isfinite(preds);a,d=auc(y[good],preds[good]);multi.append(dict(comparison=title,features=ncols,auc=a,direction=d,n0=int(sum(y[good]==0)),n1=int(sum(y[good]==1)),skipped=skip,C=str(selected),method='LOGO'))
    save('source_same_model_auc.csv',records);return records,multi
def main():
    started=time.time();df,xraw=load();valid=(df.keep=='True')&~df.meta_name.str.contains('rejected');nan=[]
    for (dataset,label),ids in df.groupby(['dataset','label']).groups.items():
        for j,n in enumerate(NAMES):nan.append(dict(dataset=dataset,label=label,feature=n,n=len(ids),nan=int(np.isnan(xraw[ids,j]).sum()),inf=int(np.isinf(xraw[ids,j]).sum())))
    save('nonfinite_counts.csv',nan);x=xraw.copy()
    for band in df.band.unique():
        ids=np.flatnonzero(df.band==band);tr=(df.iloc[ids].split=='train').values&valid.iloc[ids].values;x[ids],_=fill(x[ids],tr)
    ysrc=(df.dataset=='v3').astype(int);a=valid&(df.band=='2.4GHz')&(df.label=='background')&((df.dataset=='dataA_mirror')|((df.dataset=='v3')&df.group_id.str.contains('20260710')))
    ap=valid&(df.band=='2.4GHz')&(df.label=='background');source=univ(x,a,ysrc,'a_all')+univ(x,ap,ysrc,'aprime_all')
    for split in ['train','val','test']:source+=univ(x,a&(df.split==split),ysrc,'a_'+split)
    flags={r['feature']:('接收机指纹' if r['auc']>=.9 else '可疑' if r['auc']>=.75 else '未见来源线索') for r in source if r['comparison']=='a_all'}
    for r in source:r['classification']=flags[r['feature']]
    save('source_background_auc.csv',source);multi=[]
    for n in [13,19]:
        pp,C,_=trainlr(xraw,ysrc.values,(a&(df.split=='train')).values,(a&(df.split=='val')).values,list(range(n)));te=(a&(df.split=='test')).values;v,d=auc(ysrc[te],pp[te]);multi.append(dict(comparison='a_test',features=n,auc=v,direction=d,n0=int(sum(ysrc[te]==0)),n1=int(sum(ysrc[te]==1)),skipped=0,C=C,method='train/val/test'))
    same,smulti=source_b(df,xraw);multi+=smulti;save('source_multivariate_auc.csv',multi)
    label=[]
    for band in ['2.4GHz','5.8GHz']:
        for receiver in ['A','B']:
            if receiver=='A' and band=='5.8GHz':continue
            mask=valid&(df.band==band)&(df.dataset.isin(['data','dataA_mirror']) if receiver=='A' else (df.dataset=='v3'))
            for split in ['train','val','test']:label+=univ(x,mask&(df.split==split),(df.label=='drone').astype(int),f'{band}:{receiver}:{split}')
    save('label_auc.csv',label)
    noise=np.load(CACHE/'noise_features.npy');nd=pd.DataFrame(rows(CACHE/'noise_meta.csv'));na=[];noise_nonfinite=[]
    for (band,dataset,label0),ii in nd.groupby(['band','dataset','label']).groups.items():
        for level,snr0 in enumerate(['clean','0','-5','-10','-15']):
            for j,n in enumerate(NAMES):noise_nonfinite.append(dict(band=band,dataset=dataset,label=label0,snr=snr0,feature=n,n=len(ii),nan=int(np.isnan(noise[ii,level,j]).sum()),inf=int(np.isinf(noise[ii,level,j]).sum())))
    save('noise_nonfinite_counts.csv',noise_nonfinite)
    for band in ['2.4GHz','5.8GHz']:
        train=(valid&(df.band==band)&(df.split=='train')).values;med=np.nanmedian(xraw[train],axis=0);med=np.nan_to_num(med)
        for level,snr in enumerate(['clean','0','-5','-10','-15']):
            nx=np.where(np.isfinite(noise[:,level,:]),noise[:,level,:],med)
            for receiver in ['A','B']:
                if receiver=='A' and band=='5.8GHz':continue
                m=(nd.band==band)&(nd.dataset.isin(['data','dataA_mirror']) if receiver=='A' else (nd.dataset=='v3'));na+=univ(nx,m,(nd.label=='drone').astype(int),f'{band}:{receiver}',dict(snr=snr))
    save('noise_auc.csv',na)
    wide=[]
    for comparison in sorted(set(r['comparison'] for r in na)):
        for feature_name in NAMES:
            rr=[next(r for r in na if r['comparison']==comparison and r['feature']==feature_name and r['snr']==s) for s in ['clean','0','-5','-10','-15']]
            wide.append(dict(comparison=comparison,feature=feature_name,n0=rr[0]['n0'],n1=rr[0]['n1'],**{r['snr']:r['auc'] for r in rr},**{r['snr']+'_direction':r['direction'] for r in rr}))
    save('noise_auc_wide.csv',wide)
    base=[];fitinfo=[];rejected=[]
    for band in ['2.4GHz','5.8GHz']:
        ids=np.flatnonzero(valid&(df.band==band));dd=df.iloc[ids].reset_index(drop=True);xx=xraw[ids];y=(dd.label=='drone').astype(int).values;tr=(dd.split=='train').values;va=(dd.split=='val').values;te=(dd.split=='test').values
        weights=np.zeros(len(dd))
        masks=[(y==1,1),((dd.dataset=='v3').values&(y==0),.5 if band=='2.4GHz' else 1)]
        if band=='2.4GHz':masks.append(((dd.dataset=='dataA_mirror').values,.5))
        for m,w in masks:weights[tr&m]=w/sum(tr&m)
        weights[tr]/=np.mean(weights[tr]);safe=[i for i in range(13) if flags[NAMES[i]]=='未见来源线索']
        for title,cols in [('LR13',list(range(13))),('LR_safe',safe)]:
            pp,C,empty=trainlr(xx,y,tr,va,cols,weights);base+=metrics(dd[te],pp[te],band,title)
            save(f'predictions_{band}_{title}.csv',[dict(r,p_drone=float(p),pred_argmax=int(p>=.5)) for r,p in zip(dd[te].to_dict('records'),pp[te])]);fitinfo.append(dict(band=band,model=title,C=C,features=';'.join(NAMES[i] for i in cols),ntrain=int(tr.sum()),nval=int(va.sum()),ntest=int(te.sum()),allnan_train_features=empty))
            # Rejected test appendix uses same fitted model, no retraining to those rows.
            reject=np.flatnonzero((df.band==band)&df.meta_name.str.contains('rejected'))
            if len(reject):
                comb=np.concatenate([xx,xraw[reject]]);yy=np.r_[y,(df.iloc[reject].label=='drone').astype(int)];tr2=np.r_[tr,np.zeros(len(reject),bool)];va2=np.r_[va,np.zeros(len(reject),bool)];w2=np.r_[weights,np.zeros(len(reject))];pr,_,_=trainlr(comb,yy,tr2,va2,cols,w2);rejected+=metrics(df.iloc[reject],pr[-len(reject):],band,title)
        for seed in [2026,2027,2028]:
            folder=ROOT/'work'/('train_stage1_L20_bgA_20261109' if band=='2.4GHz' else 'train_stage1_L20_20261108');rr=rows(folder/f'test_predictions_{band}_seed{seed}.csv')
            if band=='2.4GHz':rr+= [r for r in rows(folder/f'bgA_test_predictions_seed{seed}.csv') if r.get('keep','True')=='True']
            ref=pd.DataFrame(rr);base+=metrics(ref,np.array([float(r['p_drone']) for r in rr]),band,('J' if band=='2.4GHz' else 'H')+str(seed))
    save('classifier_metrics.csv',base);save('classifier_fits.csv',fitinfo);save('rejected_classifier_metrics.csv',rejected)
    # Plot every feature, panels per receiver; all curves use complete selected test subset.
    for band in ['2.4GHz','5.8GHz']:
        fig,axes=plt.subplots(1,2 if band=='2.4GHz' else 1,figsize=(15,6),squeeze=False)
        for ax,rec in zip(axes[0],['A','B'] if band=='2.4GHz' else ['B']):
            for j,n in enumerate(NAMES):ax.plot(range(5),[next(r['auc'] for r in na if r['comparison']==f'{band}:{rec}' and r['feature']==n and r['snr']==s) for s in ['clean','0','-5','-10','-15']],label=n,ls='-' if j<13 else '--')
            counts=next(r for r in na if r['comparison']==f'{band}:{rec}')
            ax.set(xticks=range(5),xticklabels=['干净','0','−5','−10','−15'],xlabel='SNR (dB)',ylabel='标签 AUC（取对称值）',title=f"{band} 接收机 {rec}（背景 n={counts['n0']}，无人机 n={counts['n1']}）",ylim=(.48,1.02));ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/f'fig_B1_auc_vs_snr_{band}.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(22,8));aa=[next(r['auc'] for r in source if r['comparison']=='a_all' and r['feature']==n) for n in NAMES];bb=[next(r['auc'] for r in label if r['comparison']=='2.4GHz:A:test' and r['feature']==n) for n in NAMES]
    src_counts=next(r for r in source if r['comparison']=='a_all');lab_counts=next(r for r in label if r['comparison']=='2.4GHz:A:test')
    for ax,values,title in [(axes[0],aa,f"2.4G 背景来源区分\nbgA n={src_counts['n0']}，v3 n={src_counts['n1']}"),(axes[1],bb,f"接收机 A 内标签区分（test）\n背景 n={lab_counts['n0']}，无人机 n={lab_counts['n1']}")]:
        ax.barh(range(19),values,color=['#3a86b8']*13+['#e39648']*6);ax.set(yticks=range(19),yticklabels=NAMES,xlim=(.5,1.01),xlabel='AUC（取对称值）',title=title);ax.invert_yaxis();ax.axvline(.75,color='gray',ls='--');ax.axvline(.9,color='red',ls='--')
    for n in ['shape_factor','c42n','spec_flatness','duty']:
        axes[2].plot(range(5),[next(r['auc'] for r in na if r['comparison']=='2.4GHz:A' and r['feature']==n and r['snr']==s) for s in ['clean','0','-5','-10','-15']],'-o',label=n)
    nc=next(r for r in na if r['comparison']=='2.4GHz:A')
    axes[2].set(xticks=range(5),xticklabels=['干净','0','−5','−10','−15'],xlabel='SNR (dB)',ylabel='标签 AUC',title=f"接收机 A：加噪后标签区分\n背景 n={nc['n0']}，无人机 n={nc['n1']}",ylim=(.48,1.02));axes[2].legend();fig.tight_layout();fig.savefig(OUT/'图15_手工特征的来源与标签区分力.png',dpi=180);plt.close(fig)
    report(df,source,same,multi,label,na,base,fitinfo,nan,flags,time.time()-started)
def report(df,source,same,multi,label,na,base,fits,nan,flags,seconds):
    ck=json.loads((OUT/'checks.json').read_text(encoding='utf-8'));ex=json.loads((OUT/'extraction.json').read_text(encoding='utf-8'));nr=json.loads((OUT/'noise_runtime.json').read_text(encoding='utf-8'));syn=rows(OUT/'check_synthetic.csv')
    def fmt(r):return f"{r['auc']:.3f}({r['n0']}/{r['n1']},{r['direction']})"
    invalid=any(r['auc']<.9 for r in source if r['comparison']=='a_all' and r['feature'] in NAMES[13:15])
    lines=['来源检查无效' if invalid else '# 做法 B 阶段 1：特征与来源检查',f"实现检查：双实现 {ck['n_real']} 片段×19，最大绝对误差 {ck['max_abs']:.3g}，最大相对误差 {ck['max_rel']:.3g}，全部满足阈值；行对齐随机核对 {ex['aligned']} 行，通过。",'5.8G 无 bgA，双实现该组不适用；共 21 个真实片段。Hann 采用对称 np.hanning；突发数仅计相邻 0→1，不把片段起点计为上升沿。']
    for band in ['2.4GHz','5.8GHz']:
        ss=[r for r in syn if r['band']==band]
        lines.append(f"合成 {band}（各N={1228800 if band=='2.4GHz' else 2000000}）："+'；'.join(f"{r['signal']} shape={float(r['shape_factor']):.6f},c42n={float(r['c42n']):.6f},c20={float(r['c20n_abs']):.6f},flat={float(r['spec_flatness']):.6f},duty={float(r['duty']):.3f}" for r in ss)+'。')
    lines+=['缓存：E:/fusionB_features_20261110/；每份原始 meta 行序完整保留，包括 bgA 全部5630行、两频段 rejected；列名 feature_names.json。','NaN/Inf（计数按特征与 dataset,label 见 nonfinite_counts.csv；缓存保留NaN，分析仅以相应训练集特征中位数填补）：']
    for dataset,label0 in sorted(set((r['dataset'],r['label']) for r in nan)):
        rr=[r for r in nan if r['dataset']==dataset and r['label']==label0];non=[f"{r['feature']}={r['nan']}" for r in rr if r['nan']];lines.append(f"- {dataset}/{label0}：NaN "+(', '.join(non) if non else '全部19列均0')+f"；Inf={sum(r['inf'] for r in rr)}。")
    lines+=['来源单特征 AUC：括号为 n0/n1、方向；a组0=bgA、组1=v3实验室背景，b组0=data、组1=v3；方向均为填补后数值。','|特征|a 全split / 标记|a test|a′ 全背景|b 2.4 Mini / 5.8 Mini / 2.4 EVO（合并）|','|---|---|---|---|---|']
    for n in NAMES:
        rs={r['comparison']:r for r in source if r['feature']==n};bs=[r for r in same if r['feature']==n];lines.append('|'+n+'|'+fmt(rs['a_all'])+'/'+flags[n]+'|'+fmt(rs['a_test'])+'|'+fmt(rs['aprime_all'])+'|'+'; '.join(r['comparison']+'='+fmt(r) for r in bs)+'|')
    lines+=['带宽分层 b 结果及 JSON 来源见 source_same_model_auc.csv、v3_declared_bw.csv；unknown/null 不推断。5.8G 来源均标记“未能检验”。','来源多变量（19列仅来源诊断；标签模型仅用候选）：'+'；'.join(f"{r['comparison']}/{r['features']}列={fmt(r)}" for r in multi if r['comparison']=='a_test')+'。','b多变量 LOGO 与每折 C、跳过数见 source_multivariate_auc.csv；外层留出整组，C优先用剩余组的原val选，否则嵌套LOGO；内层不可估时C=1，单类别外层训练不拟合。','同接收机标签 AUC（组0背景/组1无人机）：完整train/val/test见 label_auc.csv；此表列 test。','|特征|2.4 A|2.4 B|5.8 B|','|---|---|---|---|']
    for n in NAMES:lines.append('|'+n+'|'+'|'.join(fmt(next(r for r in label if r['feature']==n and r['comparison']==c)) for c in ['2.4GHz:A:test','2.4GHz:B:test','5.8GHz:B:test'])+'|')
    lines+=['噪声逐特征、逐SNR、n0/n1详表 noise_auc.csv；图 fig_B1_auc_vs_snr_2.4GHz.png、fig_B1_auc_vs_snr_5.8GHz.png；5.8G A没有背景。','SNR定义=原片段平均功率/新增噪声功率；每片段seed=SHA256("2026|sample_id")低64位小端，按SNR序号派生SeedSequence；逐片种子和抽样见缓存 noise_meta.csv。','|接收机/特征（n0/n1）|clean|0|−5|−10|−15|','|---|---|---|---|---|---|']
    for c in ['2.4GHz:A','2.4GHz:B','5.8GHz:B']:
        for n in ['shape_factor','c42n','spec_flatness','duty']:
            rr=[next(r for r in na if r['comparison']==c and r['feature']==n and r['snr']==s) for s in ['clean','0','-5','-10','-15']];lines.append(f"|{c}/{n}({rr[0]['n0']}/{rr[0]['n1']})|"+'|'.join(f"{r['auc']:.3f}" for r in rr)+'|')
    lines+=['分类器：完整并列表 classifier_metrics.csv（两LR及J/H各2026/2027/2028；每个v3背景会话、bgA、nonG/G/含G、data/v3和所有型号召回；虚警均含k/N、单侧CP95上界），拟合参数 classifier_fits.csv，逐片预测 predictions_*.csv；筛出片段附表 rejected_classifier_metrics.csv。','分类器关键数：虚警为 k/N[CP95上界]，漏检为 k/N；所有阈值p≥0.5。']
    for band in ['2.4GHz','5.8GHz']:
        chunks=[]
        for model in ['LR13','LR_safe']+[(('J' if band=='2.4GHz' else 'H')+str(s)) for s in [2026,2027,2028]]:
            rr=[r for r in base if r['band']==band and r['model']==model];parts=[]
            for g in ['v3_all','bgA_all','nonG','includingG']:
                r=next((r for r in rr if r['group']==g),None)
                if r:parts.append(f"{g}={r['k']}/{r['n']}"+(f"[{r['cp95_upper']:.4f}]" if r['metric']=='FA' else ''))
            chunks.append(model+':'+','.join(parts))
        lines.append(band+'：'+'；'.join(chunks)+'。')
    lines+=['“未见来源线索”候选：'+','.join(n for n in NAMES[:13] if flags[n]=='未见来源线索')+'；两频段使用同一(a)筛选清单，5.8G为迁移筛选，未检验其接收机来源。','小事项：特征分析保留G并标明；分类器主漏检排除G；LR C按val原向AUC最大选、并列取较小C；标准化不加样本权重，权重仅用于LR拟合；原有时频图预测直接读CSV，不重推。',f"耗时：全量特征{ex['seconds']:.1f}s，加噪{nr['seconds']:.1f}s，分析{seconds:.1f}s；逻辑IQ读取{(ex['read_bytes']+nr['read_bytes'])/1e9:.3f}GB（不含OS预读及检查21片）；缓存大小见交付检查。",'保护目录：protected_before.json 与 protected_after.json 必须逐文件SHA256完全一致；不导入crop.py/common.py，不写H缓存，无网络训练/安装/提交/DSH。','**事实陈述，不下结论**',f"(a)阳性对照功率/底噪 AUC："+'/'.join(f"{r['auc']:.6f}(n0={r['n0']},n1={r['n1']})" for r in source if r['comparison']=='a_all' and r['feature'] in NAMES[13:15])+'。',f"候选13列：(a)接收机指纹={sum(flags[n]=='接收机指纹' for n in NAMES[:13])}，可疑={sum(flags[n]=='可疑' for n in NAMES[:13])}，未见来源线索={sum(flags[n]=='未见来源线索' for n in NAMES[:13])}。"]
    # Keep the requested short report, while detailed CSVs retain every comparison.
    for i,line in enumerate(lines):
        if line.startswith('缓存：'):
            lines[i]+=' 汇总图：[图15](图15_手工特征的来源与标签区分力.png)。'
        if line.startswith('耗时：'):
            lines[i]=line.replace('缓存大小见交付检查',f"缓存合计{sum(p.stat().st_size for p in CACHE.iterdir() if p.is_file())/1e6:.3f}MB")
        if line.startswith('保护目录：') and (OUT/'protected_after.json').exists():
            before=json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'))
            after=json.loads((OUT/'protected_after.json').read_text(encoding='utf-8'))
            assert before==after
            lines[i]=line.replace('必须逐文件SHA256完全一致',f"已核对{len(before)}个文件SHA256完全一致")
        if line.startswith('小事项：'):
            lines[i]+=' 异常已修复：混合bgA/原始meta列不同致CSV首次写出中止，保留已完成数组并按同种子恢复2462行元数据、全行clean特征精确核对；报告变量重名修正。加噪耗时由数组完成时间减原始阶段结束时间估算，含阶段启动开销。'
        if line.startswith('NaN/Inf（'):
            nr_counts=rows(OUT/'noise_nonfinite_counts.csv')
            lines[i]+=f" 加噪缓存逐SNR分组计数见 noise_nonfinite_counts.csv，NaN合计{sum(int(r['nan']) for r in nr_counts)}、Inf合计{sum(int(r['inf']) for r in nr_counts)}。"
        if line.startswith('分类器关键数：'):
            lines[i]='|频段（虚警k/N[CP95]；漏检k/N）|LR13|LR_safe|J/H2026|J/H2027|J/H2028|\n|---|---|---|---|---|---|'
        if line.startswith(('2.4GHz：','5.8GHz：')):
            band,detail=line.split('：',1)
            lines[i]='|'+band+'|'+'|'.join(detail.rstrip('。').split('；'))+'|'
        if line.startswith('噪声逐特征、逐SNR'):
            lines[i]=line.replace('noise_auc.csv','noise_auc.csv（长表）、noise_auc_wide.csv（每个SNR一列）')
        if line.startswith('带宽分层 b 结果'):
            lines[i]+=' pooled为全部合并；unmatched为其余带宽/unknown，单类时AUC不可估。分层数值：'+'；'.join(r['comparison']+'='+fmt(r) for r in same if r['feature']=='shape_factor' and not r['comparison'].endswith(':pooled'))+'。'
            bwrows=rows(OUT/'v3_declared_bw.csv')
            lines[i]+=' declared.bw列表取唯一非空Hz值换算M；unknown组：'+','.join(r['group_id'] for r in bwrows if r['bw']=='unknown')+'。b主分析仅含keep=True，rejected仅列附表。'
        if line.startswith('b多变量 LOGO'):
            lines[i]+=' 分层Mini每档v3仅1个group，EVO的data仅1个group；留出该group后训练仅单类，对应双类OOF AUC不可估，未改换划分。'
        if line.startswith('|特征|a 全split'):
            lines[i]=line.replace('b 2.4 Mini / 5.8 Mini / 2.4 EVO（合并）','b 同型号（pooled及带宽分层）')
        if line.startswith('b多变量 LOGO'):
            lines[i]+=' AUC：'+'；'.join(f"{r['comparison']}/{r['features']}列={fmt(r)}" for r in multi if r['method']=='LOGO')+'。'
    rendered='\n'.join(lines)+'\n';nlines=len(rendered.splitlines())
    for band in ['2.4GHz','5.8GHz']:
        facts=[r for r in base if r['band']==band and r['model']=='LR13' and r['group'] in ['v3_all','nonG']]
        rendered+=band+' LR13：'+'；'.join(f"{r['group']}={r['k']}/{r['n']}" for r in facts)+'。\n'
    nlines=len(rendered.splitlines())
    assert nlines<=90,nlines;(OUT/'REPORT.md').write_text(rendered,encoding='utf-8');dump(OUT/'analysis_runtime.json',dict(seconds=seconds,report_lines=nlines,flags=flags))
if __name__=='__main__':main()
