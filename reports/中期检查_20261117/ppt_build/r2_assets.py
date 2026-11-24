from pathlib import Path
from PIL import Image
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[2];A=B/'r2_assets';A.mkdir(exist_ok=True);F=B/'r2_refs'
log=[]
def crop(src,name,rect,scale=1):
 im=Image.open(src);rect=tuple(int(v*scale) for v in rect);im.crop(rect).save(A/name);log.append({'source':str(src),'output':name,'crop':rect})
crop(F/'wisig200-06.png','wisig_fig10.png',(870,145,1555,515))
crop(F/'gluge200-03.png','gluge_mix.png',(515,675,950,1115),200/120)
crop(F/'gluge200-03.png','gluge_mix3.png',(515,998,950,1115),200/120)
crop(F/'openrf200-05.png','openrf_concat.png',(295,87,781,275),200/120)
crop(F/'drone200-07.png','drone_table4.png',(80,131,478,436),200/120)
crop(F/'soltani200-04.png','soltani97.png',(515,1065,945,1245),200/120)
crop(F/'soltani200-05.png','soltani49.png',(80,90,505,240),200/120)
crop(R/'reports/结果图表/06_验证实验_挪动和遮挡信号.png','move.png',(0,45,1500,630))
crop(R/'reports/结果图表/06_验证实验_挪动和遮挡信号.png','move_focus.png',(250,48,873,154))
crop(R/'reports/结果图表/07_时频图_Matrice300各文件片段.png','channel_before.png',(0,190,495,370))
crop(R/'reports/结果图表/07_时频图_Matrice300各文件片段.png','channel_after.png',(0,375,495,554))
crop(R/'reports/数据筛选_20261013/全文件时频图/T0100_D00_S0000.png','m100.png',(0,0,1540,343))
crop(R/'reports/数据筛选_20261013/模型输入时频图示例_15ms.png','m100_window.png',(0,865,610,1060))
crop(R/'work/prep_lines_20261104/lines_data_2.4GHz_page10.png','line2426.png',(0,3864,1820,4186))
crop(B.parent/'figs/图C_加入接收机A背景前后.png','bgA_fa.png',(0,65,1285,960))
crop(R/'work/fusionB_20261110/B1b/图17_接收机A与B镜像背景对比.png','mirror_feature.png',(1810,590,2470,940))
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':18,'axes.spines.top':False,'axes.spines.right':False})
mr=list(csv.DictReader(open(R/'work/side_L_wifimix_20261110/metrics_paired.csv',encoding='utf-8-sig')))
fig,axs=plt.subplots(1,2,figsize=(7,4.2))
for j,g in enumerate(['J','La','Lb']):
 cr=[r for r in mr if r['category']=='C1' and r['mode']=='argmax'];vr=next(r for r in mr if r['category']=='main' and r['group']=='v3_drone_recall' and r['mode']=='argmax')
 vals=[sum(float(r[f'{g}_{s}_k']) for r in cr)/sum(float(r[f'{g}_{s}_n']) for r in cr)*100 for s in [2026,2027,2028]]
 axs[0].bar(g,np.mean(vals),yerr=np.std(vals,ddof=1),capsize=4,color=['#A6BCCB','#F0A30A','#005A8C'][j]);axs[1].bar(g,100*float(vr[g+'_mean']),yerr=100*float(vr[g+'_std']),capsize=4,color=['#A6BCCB','#F0A30A','#005A8C'][j])
axs[0].set_title('叠加集虚警',fontsize=17);axs[1].set_title('城区无人机召回',fontsize=17)
for ax in axs:ax.set_ylabel('比例（%）');ax.grid(axis='y',alpha=.15)
fig.tight_layout();fig.savefig(A/'wifi_focus.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.2));vals=[0,93.2,.9];ax.bar(['背景虚警','实验室检出','城区检出'],vals,color=['#A6BCCB','#0079BA','#F0A30A']);ax.set_ylim(0,108);ax.set_ylabel('比例（%）')
for j,v in enumerate(vals):ax.text(j,v+3,f'{v:g}%',ha='center')
fig.tight_layout();fig.savefig(A/'source_probe.png',dpi=200);plt.close(fig)
log.append({'source':'R2任务书第12页','output':'source_probe.png','data':[0,93.2,.9],'note':'同来源0.51对照在正文说明，不混入百分比坐标'})
rows=list(csv.DictReader(open(R/'work/main_W_window_20261110/metrics_summary.csv',encoding='utf-8-sig')))
for group,lab,name in [('nonG_miss','实验室 nonG 漏检率（%）','window_miss.png'),('v3_drone_recall','城区无人机召回率（%）','window_recall.png')]:
 fig,ax=plt.subplots(figsize=(6,4.3))
 for band,c in [('2.4GHz','#0079BA'),('5.8GHz','#F0A30A')]:
  rs=sorted([(i+2,r) for i,r in enumerate(rows) if r['mode']=='argmax' and r['category']=='main' and r['group']==group and r['band']==band],key=lambda x:float(x[1]['L'].replace('t','')))
  x=[float(r['L'].replace('t','')) for _,r in rs];y=[100*float(r['mean']) for _,r in rs]
  ax.errorbar(x,y,yerr=[100*float(r['std']) for _,r in rs],color=c,marker='o',capsize=3,label=band)
  for (i,r),xx,yy in zip(rs,x,y):
   rr=next((j+2,z) for j,z in enumerate(rows) if z['L']==r['L'] and z['band']==band and z['mode']=='argmax' and z['category']=='main' and z['group']=='v3_test_FA')
   ax.annotate('N='+rr[1]['n'],(xx,yy),xytext=(0,11 if band=='2.4GHz' else -23),textcoords='offset points',fontsize=11,ha='center',color=c)
   log.append({'source':str(R/'work/main_W_window_20261110/metrics_summary.csv'),'row':i,'N_row':rr[0],'output':name,'data':r})
 ax.set_xlabel('窗长（ms）');ax.set_xticks([10,20,30,40,50]);ax.set_ylabel(lab);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=12);ax.margins(y=.25);fig.tight_layout();fig.savefig(A/name,dpi=200);plt.close(fig)
(A/'manifest.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print('R2 assets ready')
