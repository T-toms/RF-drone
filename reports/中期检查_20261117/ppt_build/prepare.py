import os,csv,json
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).resolve().parent; R=D.parents[2]; F=D.parent/'figs'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':17,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#404040','text.color':'#404040'})
assets=[]; provenance=[]
def crop(src,name,rect):
 im=Image.open(src); im.crop(rect).save(D/name); assets.append([str(src),name,str(rect)])
crop(F/'图B_背景替换对照.png','background.png',(0,105,1800,682))
crop(F/'图C_加入接收机A背景前后.png','bgA.png',(0,65,2560,960))
crop(F/'图E_时频图加IQ特征对比.png','fusion_fa.png',(0,75,1530,900))
crop(F/'图E_时频图加IQ特征对比.png','fusion_recall.png',(1530,925,3040,1810))
Q=R/'work/side_Q2_band_detect_20261111'
crop(Q/'图_k6_g2_data_drone_01.png','normal.png',(785,25,1530,285))
crop(Q/'图_k6_g2_raw_background_01.png','merged.png',(785,550,1530,810))
crop(R/'work/prep_lines_20261104/lines_data_2.4GHz.png','lines.png',(0,0,Image.open(R/'work/prep_lines_20261104/lines_data_2.4GHz.png').width,Image.open(R/'work/prep_lines_20261104/lines_data_2.4GHz.png').height))
def load(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
W=R/'work/main_W_window_20261110/metrics_summary.csv'; rows=load(W)
fig,axs=plt.subplots(2,1,figsize=(9,7.2),sharex=True)
for ax,group,title in zip(axs,['nonG_miss','v3_drone_recall'],['nonG 漏检率（%）','城区 v3 无人机召回率（%）']):
 for b,c in [('2.4GHz','#0079BA'),('5.8GHz','#F0A30A')]:
  rs=sorted([(i+2,r) for i,r in enumerate(rows) if r['mode']=='argmax' and r['category']=='main' and r['group']==group and r['band']==b],key=lambda x:float(x[1]['L'].replace('t','')))
  ax.errorbar([float(r['L'].replace('t','')) for _,r in rs],[100*float(r['mean']) for _,r in rs],yerr=[100*float(r['std']) for _,r in rs],label=b,color=c,marker='o',capsize=4,lw=2)
  for i,r in rs:provenance.append(['窗长图',str(W),i,'L,band,group,mean,std',r])
 ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
axs[0].legend(ncol=2,loc='upper right');axs[1].set_xlabel('窗长（ms）');axs[1].set_xticks([10,20,30,40,50]);fig.tight_layout();fig.savefig(D/'window.png',dpi=220);plt.close(fig)
P=R/'work/side_L_wifimix_20261110/metrics_paired.csv';rows=load(P)
spec=[('C1','all','叠加集虚警'),('main','v3_test_FA','城区背景虚警'),('main','v3_drone_recall','城区无人机召回'),('main','data_nonG_recall','实验室非G召回')]
fig,axs=plt.subplots(1,2,figsize=(15,4))
for panel in range(2):
 ax=axs[panel]
 for j,g in enumerate(['J','La','Lb']):
  vals=[];errs=[]
  for cat,sub,label in spec[panel*2:panel*2+2]:
   matches=[(i+2,r) for i,r in enumerate(rows) if r['category']==cat and r['group']==sub and r['mode']=='argmax']
   if not matches and cat=='C1':matches=[(i+2,r) for i,r in enumerate(rows) if r['category']=='C1' and r['mode']=='argmax' and r['group'] in ['all','pooled','merged','C1_all']]
   if not matches and cat=='C1':
    rs=[(i+2,r) for i,r in enumerate(rows) if r['category']=='C1' and r['mode']=='argmax']
    v=[sum(float(r[f'{g}_{s}_k']) for _,r in rs)/sum(float(r[f'{g}_{s}_n']) for _,r in rs)*100 for s in [2026,2027,2028]]
    vals.append(np.mean(v));errs.append(np.std(v,ddof=1))
    for i,r in rs:provenance.append(['WiFi图C1七档合并',str(P),i,g+'各种子k/n，先合并七档再均值和样本标准差',r])
    continue
   if not matches: raise ValueError((cat,sub))
   i,r=matches[0];vals.append(100*float(r[g+'_mean']));errs.append(100*float(r[g+'_std']));provenance.append(['WiFi图',str(P),i,g+'_mean,'+g+'_std',r])
  ax.bar(np.arange(2)+(j-1)*.24,vals,.24,yerr=errs,capsize=3,label=g,color=['#A6BCCB','#F0A30A','#005A8C'][j])
 ax.set_xticks([0,1],[x[2] for x in spec[panel*2:panel*2+2]]);ax.set_ylabel('比例（%）');ax.set_ylim(0,110 if panel else 45);ax.grid(axis='y',alpha=.15);ax.legend(ncol=3)
fig.tight_layout();fig.savefig(D/'wifi.png',dpi=220);plt.close(fig)
(D/'assets.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2),encoding='utf-8')
(D/'chart_sources.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
FP=R/'work/fusionB_B2_20261112/metrics_summary.csv';rs=load(FP)
fig,axs=plt.subplots(2,1,figsize=(9,7.2))
for ax,band,subset,label in zip(axs,['2.4GHz','5.8GHz'],['v3_test_FA','v3_drone_recall'],['2.4 GHz 城区背景虚警（%）','5.8 GHz 城区无人机召回（%）']):
 vals=[];errs=[]
 for g in ['R','Z','F','F5']:
  i,r=next((i+2,r) for i,r in enumerate(rs) if r['band']==band and r['group']==g and r['subset']==subset and r['mode']=='argmax')
  vals.append(float(r['rate_mean'])*100);errs.append(float(r['rate_std'])*100);provenance.append(['融合图',str(FP),i,'rate_mean,rate_std',r])
 ax.bar(['R','Z','F','F5'],vals,yerr=errs,capsize=5,color=['#A6BCCB','#F0A30A','#0079BA','#005A8C']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(D/'fusion_focus.png',dpi=220);plt.close(fig)
(D/'chart_sources.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
print('assets ready')
