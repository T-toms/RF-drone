from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
B=Path(__file__).resolve().parent
src=(B/'r2_assets.py').read_text(encoding='utf-8-sig').replace("A=B/'r2_assets'","A=B/'r3_assets'")
src=src.replace("axs[0].bar(g,","axs[0].bar(['不叠加','只给无人机\\n叠加','对称叠加'][j],").replace("axs[1].bar(g,","axs[1].bar(['不叠加','只给无人机\\n叠加','对称叠加'][j],")
src=src.replace('叠加集虚警','叠加测试集误报率').replace('城区无人机召回','城区无人机检出率').replace('背景虚警','背景误报率').replace('实验室 nonG 漏检率','实验室漏检率').replace('城区无人机召回率','城区无人机检出率')
src=src.replace('figsize=(7,4.2)','figsize=(11,4.8)')
src=src.replace("fig.tight_layout();fig.savefig(A/'wifi_focus.png'","[ax.tick_params(axis='x',labelsize=13) for ax in axs];fig.tight_layout();fig.savefig(A/'wifi_focus.png'")
exec(src,globals())
import csv
rs=list(csv.DictReader(open(R/'work/fusionB_B2_20261112/metrics_summary.csv',encoding='utf-8-sig')))
fig,axs=plt.subplots(1,2,figsize=(11,4.8));labs=['只用时频图','特征全置零','10个特征','5个特征'];record=[]
for ax,band,subset,title in zip(axs,['2.4GHz','5.8GHz'],['v3_test_FA','v3_drone_recall'],['城区背景误报率','城区无人机检出率']):
 vals=[]
 for i,g in enumerate(['R','Z','F','F5']):
  ix,r=next((ix,r) for ix,r in enumerate(rs,2) if r['group']==g and r['band']==band and r['subset']==subset and r['mode']=='argmax')
  v=100*float(r['rate_mean']);vals.append(v);ax.bar(i,v,yerr=100*float(r['rate_std']),capsize=4,color=['#CFE3F3','#0079BA','#79B4DB','#005A8C'][i]);ax.text(i,v+3,f'{v:.1f}%',ha='center',fontsize=13)
  record.append({'source':str(R/'work/fusionB_B2_20261112/metrics_summary.csv'),'row':ix,'value':v,'std':100*float(r['rate_std'])})
 ax.set_xticks(range(4));ax.set_xticklabels(labs,fontsize=12);ax.set_title(band+' '+title,fontsize=17);ax.set_ylabel('比例（%）');ax.set_ylim(0,max(vals)*1.3+2);ax.grid(axis='y',alpha=.15)
 if band=='5.8GHz':ax.annotate('',xy=(3,vals[3]),xytext=(1,vals[1]),arrowprops={'arrowstyle':'->','color':'#F0A30A','lw':2})
fig.tight_layout();fig.savefig(A/'fusion.png',dpi=300);plt.close(fig)
(A/'fusion_sources.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
