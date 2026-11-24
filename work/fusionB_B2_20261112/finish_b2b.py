from finish import oldrefs,plot,extras,auxiliary
from evaluate import *
GROUPS=['R','Z','F','F5']
def supplemental():
 from common import image
 rr=rows(OUT/'metrics_summary.csv')
 for b in ['2.4GHz','5.8GHz']:
  x,f,rs=loadset(b+'_test');fig,axs=plt.subplots(2,3,figsize=(15,8))
  for col,(ds,lab) in enumerate([('data','drone'),('v3','drone'),('v3','background')]):
   ids=[i for i,r in enumerate(rs) if r['dataset']==ds and r['label']==lab]
   for ax,i in zip(axs[:,col],ids[:2]):image(ax,x[i],b,ds+' '+lab+'\n'+rs[i]['sample_id'])
  fig.tight_layout();fig.savefig(OUT/f'fig0_input_examples_{b}.png',dpi=130);plt.close(fig)
  for mode in ['argmax','thr_fa1']:
   mm=[r for r in rr if r['band']==b and r['mode']==mode and r['category']=='mix_model'];models=sorted(set(' | '.join(r['subset'].split(' | ')[:-1]) for r in mm))
   fig,axs=plt.subplots(int(np.ceil(len(models)/4)),4,figsize=(22,4*int(np.ceil(len(models)/4))),squeeze=False)
   for ax,model in zip(axs.flat,models):
    for g in GROUPS:
     zz=sorted([r for r in mm if r['group']==g and ' | '.join(r['subset'].split(' | ')[:-1])==model],key=lambda r:float(r['subset'].split(' | ')[-1]))
     ax.errorbar([float(r['subset'].split(' | ')[-1]) for r in zz],[float(r['rate_mean']) for r in zz],yerr=[float(r['rate_std']) for r in zz],label=g,marker='o',capsize=2)
    ax.set(title=model,xlabel='叠加功率比 (dB)',ylabel='召回率',ylim=(-.03,1.05));ax.legend(fontsize=7)
   for ax in list(axs.flat)[len(models):]:ax.axis('off')
   fig.tight_layout();fig.savefig(OUT/f'fig_mix_by_model_{b}_{mode}.png',dpi=120);plt.close(fig)
   fig,ax=plt.subplots(figsize=(10,5))
   for j,g in enumerate(GROUPS):
    z=[next(r for r in rr if r['band']==b and r['mode']==mode and r['group']==g and r['category']=='source_recall' and r['subset']==ds) for ds in ['data','v3']]
    ax.bar(np.arange(2)+(j-1.5)*.2,[float(r['rate_mean']) for r in z],.2,yerr=[float(r['rate_std']) for r in z],label=g,capsize=3)
   ax.set_xticks([0,1],['data无人机（含G）','v3无人机（含G）']);ax.set(title=b+' '+mode,ylabel='召回率',ylim=(0,1.05));ax.legend();fig.tight_layout();fig.savefig(OUT/f'fig2_recall_by_source_{b}_{mode}.png',dpi=140);plt.close(fig)
def newplots():
 dep=rows(OUT/'dependence_all.csv');single=rows(OUT/'feature_permutation_single.csv')
 for which in ['dependence','single']:
  fig,axs=plt.subplots(1,2,figsize=(19,6))
  for ax,b in zip(axs,['2.4GHz','5.8GHz']):
   if which=='single':
    for j,g in enumerate(['F','F5']):
     rr=[r for r in single if r['band']==b and r['group']==g and r['subset']=='v3_drone_recall' and r['mode']=='argmax']
     lookup={r['feature']:r for r in rr};xs=[i for i,n in enumerate(NAMES) if n in lookup];ax.bar(np.array(xs)+(j-.5)*.35,[float(lookup[NAMES[i]]['delta_mean'])*100 for i in xs],.35,yerr=[float(lookup[NAMES[i]]['delta_std'])*100 for i in xs],label=g,capsize=3)
    ax.set_xticks(range(10),NAMES,rotation=45,ha='right');ax.set_ylabel('v3 无人机召回变化（百分点）')
   else:
    for j,(g,method) in enumerate((g,m) for g in ['F','F5'] for m in ['shuffle_all','shuffle_label','zero']):
     vals=[];errs=[]
     for sub in ['v3_drone_recall','nonG_miss']:
      zz=[float(r['delta'])*100 for r in dep if r['band']==b and r['group']==g and r['set']=='test' and r['subset']==sub and r['mode']=='argmax' and r['dependence']==method];vals.append(np.mean(zz));errs.append(np.std(zz,ddof=1))
     ax.bar(np.arange(2)+(j-2.5)*.13,vals,.13,yerr=errs,label=g+' '+{'shuffle_all':'全体打乱','shuffle_label':'标签内打乱','zero':'置零'}[method],capsize=2)
    ax.set_xticks([0,1],['v3无人机召回','非G漏检']);ax.set_ylabel('相对原模型变化（百分点）')
   ax.set_title(b);ax.axhline(0,color='black',lw=.7);ax.legend(fontsize=8)
  fig.tight_layout();fig.savefig(OUT/('fig_单特征打乱.png' if which=='single' else 'fig_feature_dependence.png'),dpi=160);plt.close(fig)
def report():
 before=json.loads((OUT/'protected_B2b_before.json').read_text(encoding='utf-8'));after=hashes();dump(OUT/'protected_B2b_after.json',after);assert before==after,'PROTECTED HASH MISMATCH'
 rr=rows(OUT/'metrics_summary.csv');raw=rows(OUT/'metrics_all.csv');models=rows(OUT/'best_models.csv');dep=rows(OUT/'dependence_all.csv');sp=rows(OUT/'feature_permutation_single.csv');pre=json.loads((OUT/'preflight.json').read_text(encoding='utf-8'));done=[json.loads(p.read_text(encoding='utf-8')) for p in OUT.glob('*_done.json') if p.name.startswith(('R_','Z_','F_','F5_')) and 'smoke' not in p.name]
 assert len(models)==len(done)==24
 def fmt(v):return f'{np.mean(v):.6f}±{np.std(v,ddof=1):.6f}'
 def rates(b,g,sub,cat='main'):
  return [float(next(r['rate'] for r in raw if r['band']==b and r['group']==g and r['subset']==sub and r['category']==cat and r['mode']=='argmax' and r['seed']==str(s))) for s in [2026,2027,2028]]
 lines=['# B2b 一阶段时频图与IQ特征融合',
 '完成24次40 epoch正式训练及4次3 epoch冒烟；协调者确认32次为笔误，按3种子×2频段×4组执行；seed2026中间结果见interim_seed2026.md/csv。',
 '设置：seed2026/2027/2028；AdamW lr1e-3、wd1e-4；cosine T_max40至1e-5；batch64；自动AMP；cudnn.benchmark=True；无增强。',
 '结构：R=ResNet18(1,2)；F/Z/F5复用stem/layer1–4/pool→512→LayerNorm；IQ 10/10/5→Linear64→ReLU→LayerNorm；拼接576→Dropout0.3→Linear2；Z标准化后置零。',
 '参数量：'+ '；'.join(g+'='+next(r['parameters'] for r in models if r['group']==g) for g in GROUPS)+'。',
 '与L/H差异：增加融合/预处理/输入分支、新缓存路径及RNG断点；保留L-b masses、验证权重、FA_v3+FA_bgA+Miss选模与双背景阈值；H保留1/count、FA+Miss选模。',
 '10列：'+', '.join(NAMES)+'；F5取第2/5/7/8/9列（0起）。',
 '协调者决定：排除6个ctl_*及c40n_abs/c41n_abs/c20n_abs（后3者对3dB以下弱内容敏感）。B1b中F5五列未见接收机差异；spec_flatness是接收机指纹，其余四列可疑；c20n_abs叠加缓存未算，本次不加。',
 '填补中位数/z-score均值和总体标准差/±10截断仅拟合本频段训练集（含叠加）；F5直接取F参数对应列；*_normalization.json及checkpoint存参数，F5 checkpoint列名为5列。']
 lines += ['异常处理：修正Orca可执行路径及Windows默认GBK读取UTF-8文件；训练未重跑；12个融合模型的图像分支复用均与完整前向差0。补充图见fig0_input_examples_*、fig2_recall_by_source_*、fig_mix_by_model_*与fig4*。']
 for b in ['2.4GHz','5.8GHz']:
  n=json.loads((OUT/(b+'_normalization.json')).read_text(encoding='utf-8'));lines.append(f"{b}训练缺失={n['train_missing']}；截断={n['train_clip']}（按10列顺序）；各集合计数见normalization_counts.csv。")
 lines += [f"沿用预检：9份meta一致；特征复算最大差{max(r['max_abs'] for r in pre['features']):.3g}；3次spec重复一致；9次叠加最大误差{max(r['delta_db'] for r in pre['mix']):.3g}dB。",'bgA经批准：先核对B1完整5630行，再按原row_index筛keep的5138行；train/val/test=2861/990/1287；未重建缓存。',f'重新开工nvidia-smi无其他Python计算；保护文件{len(before)}个SHA256前后完全一致（protected_B2b_before/after.json）。','2.4G训练计数：data无人机2832、v3无人机152、v3背景234、bgA2861、叠加无人机2832、叠加背景2861；masses=.25/.25/.125/.25/.125。','修正既有脚本：补F5和种子轮次/中间评估、显式cuDNN设置、内存统计结构；评估新增接收机/逐列置换，并复用eval图像分支；未安装包、未调用DSH、未导入crop.py。','CP95单侧上界核验：0/80=3.675420%；0/1287=0.232498%；0/990=0.302142%；1/80=5.792868%。','|频段/组/种子|best epoch|val FA_v3|val FA_bgA|val Miss|thr_fa1|','|---|---:|---:|---:|---:|---:|']
 for r in models:lines.append('|'+ '/'.join(r[k] for k in ['band','group','seed'])+'|'+ '|'.join(r[k] for k in ['epoch','val_fa_v3','val_fa_bgA','val_miss','thr_fa1'])+'|')
 lines += ['完整逐种子/均值±样本标准差、argmax/thr_fa1、虚警k/N和CP95见metrics_all/summary.csv；逐片预测R_/Z_/F_/F5_*_predictions.csv。','旧checkpoint（不是本次训练）：old_reference_metrics.csv及comparison.csv；原日志对照training_reference.csv，仅描述不作为停止条件。','图20_时频图加IQ特征_一阶段对比.png（四组）；fig_feature_dependence.png；fig_单特征打乱.png；fig3_mix_test_*.png；fig_C1_2.4GHz.png；按型号叠加指标在metrics_all.csv。','依赖度：F/F5在test/C1全体及同标签置换各5次（rng20261112）、置零一次，见dependence_all/summary.csv；delta相对各自原模型。','逐列置换：F10列/F5五列，test/bgA test/C1，rng20261116，每列5次；feature_permutation_single_raw.csv保留逐种子，feature_permutation_single.csv为3种子×5次均值±标准差。','LR10：同B1原始keep数据，无叠加；训练中位数/StandardScaler及子组权重，val AUROC选C∈[.01,.1,1,10]；*_LR10_fit.json；test指标在metrics_all.csv。','接收机A=data无人机+bgA（2.4G），B=v3无人机+v3背景；非G召回。5.8G接收机A没有背景片，接收机与标签混在一起。metrics_by_receiver.csv保留逐种子及单列对照。','|接收机摘要（argmax）|R|Z|F|F5|','|---|---|---|---|---|']
 for b in ['2.4GHz','5.8GHz']:
  for sub in (['A_FA','A_nonG_recall','B_FA','B_nonG_recall'] if b=='2.4GHz' else ['A_nonG_recall','B_FA','B_nonG_recall']):lines.append('|'+b+' '+sub+'|'+'|'.join(fmt(rates(b,g,sub,'receiver')) for g in GROUPS)+'|')
 lines += [f"正式训练累计{sum(d['seconds'] for d in done):.1f}s；训练显存峰值{max(d['peak_gpu_bytes'] for d in done)/2**30:.3f}GiB；进程内存峰值{max(d['peak_host_bytes'] for d in done)/2**30:.3f}GiB；本任务输出+缓存{sum(p.stat().st_size for d in [OUT,CACHE] for p in d.rglob('*') if p.is_file())/2**30:.3f}GiB；run.log记录过程。",'**事实陈述，不下结论**']
 for b in ['2.4GHz','5.8GHz']:
  items=[('main',n) for n in ['v3_test_FA','nonG_miss','v3_drone_recall']]+([('main',n) for n in ['bgA_test_FA','C1_FA','notched_noise_FA']]+[('v3_model_recall',n) for n in ['mavic2','mini3pro']] if b=='2.4GHz' else [])
  for cat,sub in items:lines.append(b+' '+sub+'（R→Z→F→F5）='+' → '.join(fmt(rates(b,g,sub,cat)) for g in GROUPS))
  lines.append(b+' 接收机B FA/非G召回（R→Z→F→F5）='+' → '.join(fmt(rates(b,g,'B_FA','receiver'))+'/'+fmt(rates(b,g,'B_nonG_recall','receiver')) for g in GROUPS))
  for g in ['F','F5']:
   top=sorted([r for r in sp if r['band']==b and r['group']==g and r['subset']=='v3_drone_recall' and r['mode']=='argmax'],key=lambda r:abs(float(r['delta_mean'])),reverse=True)[:3]
   lines.append(b+' '+g+' 单列置换召回变化前三：'+'；'.join(r['feature']+'='+r['delta_mean']+'±'+r['delta_std'] for r in top))
   vals=[]
   for method in ['shuffle_all','zero']:
    for sub in ['v3_drone_recall','nonG_miss']:
     v=[np.mean([float(r['delta']) for r in dep if r['band']==b and r['group']==g and r['seed']==str(s) and r['mode']=='argmax' and r['set']=='test' and r['subset']==sub and r['dependence']==method]) for s in [2026,2027,2028]];vals.append(method+' '+sub+'='+fmt(v))
   lines.append(b+' '+g+' 依赖度：'+'；'.join(vals))
   vals=[]
   for sub in ['v3_test_FA','nonG_miss','v3_drone_recall']+(['C1_FA'] if b=='2.4GHz' else []):
    v=np.array(rates(b,g,sub))-rates(b,'Z',sub);vals.append(sub+'=['+','.join(f'{z:.6f}' for z in v)+']；方向一致='+str(len(set(np.sign(v)))==1))
   lines.append(b+' '+g+'−Z配对差（seed2026/2027/2028）：'+'；'.join(vals))
 formatted=[]
 for i,line in enumerate(lines):
  if formatted and ((line.startswith('|') and not lines[i-1].startswith('|')) or (not line.startswith('|') and lines[i-1].startswith('|')) or line=='**?????????**' or lines[i-1]=='**?????????**'):formatted.append('')
  formatted.append(line)
 lines=formatted
 assert len(lines)<=100,len(lines)
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(OUT/'acceptance.json',dict(protected_identical=True,protected_files=len(before),models=24,report_lines=len(lines)))
if __name__=='__main__':oldrefs();plot();extras();auxiliary();newplots();supplemental();report()
