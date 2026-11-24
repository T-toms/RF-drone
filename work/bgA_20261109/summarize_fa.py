from pathlib import Path
import csv,json,numpy as np,collections
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
def rows(name):return list(csv.DictReader((O/name).open(encoding='utf-8-sig')))
def write(name,rs):
 with (O/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def stat(rr,rule):
 n=len(rr);k=sum(int(r['pred_'+rule]) for r in rr);return dict(N=n,positive=k,rate=k/n,cp95_upper=float(beta.ppf(.95,k+1,n-k)) if k<n else 1.)
def main():
 allbg=rows('bgA_predictions.csv');bg=[r for r in allbg if r['keep']=='True'];pair=[r for r in rows('paired_drone_predictions.csv') if r['bgA_keep']=='True'];out=[];pr=[]
 for seed in [2026,2027,2028]:
  rr=[r for r in bg if int(r['seed'])==seed];pp=[r for r in pair if int(r['seed'])==seed]
  for kind in ['split','source']:
   key='split' if kind=='split' else 'group_id'
   for value in sorted({r[key] for r in rr}):
    gr=[r for r in rr if r[key]==value];gp=[r for r in pp if r[key]==value]
    for rule in ['argmax','thr']:
     out.append(dict(kind=kind,value=value,seed=seed,rule=rule,source_model=gr[0]['source_model'] if kind=='source' else '',source_state=gr[0]['source_state'] if kind=='source' else '',source_bw_label=gr[0]['source_bw_label'] if kind=='source' else '',**stat(gr,rule)));pr.append(dict(kind=kind,value=value,seed=seed,rule=rule,**stat(gp,rule)))
 for seed in [2026,2027,2028]:
  failed=[r for r in allbg if r['keep']=='False' and int(r['seed'])==seed]
  if failed:
   for rule in ['argmax','thr']:out.append(dict(kind='failed_blocks',value='all_failed',seed=seed,rule=rule,source_model='',source_state='',source_bw_label='',**stat(failed,rule)))
 write('bgA_FA.csv',out);write('paired_recall.csv',pr)
 groups=sorted({r['group_id'] for r in bg});fig,ax=plt.subplots(figsize=(18,6));x=np.arange(len(groups));width=.25
 for j,seed in enumerate([2026,2027,2028]):
  vals=[next(r['rate'] for r in out if r['kind']=='source' and r['value']==g and r['seed']==seed and r['rule']=='argmax')*100 for g in groups];ax.bar(x+(j-1)*width,vals,width,label=str(seed))
 ax.set_xticks(x);ax.set_xticklabels([g.replace('data:','').replace('2.4GHz_','').replace('2.4GHZ_','') for g in groups],rotation=90,fontsize=6);ax.set(ylabel='背景虚警率 (%)，argmax',title='接收机A镜像代理背景：按源录制与种子');ax.legend();fig.tight_layout();fig.savefig(O/'背景FA_按源录制.png',dpi=150);plt.close(fig)
 print('FA_TABLES_DONE',len(out),flush=True)
if __name__=='__main__':main()
